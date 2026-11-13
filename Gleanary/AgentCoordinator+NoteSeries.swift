import Foundation

/// Fixed, bounded specialist stages with durable, immutable handoffs. Only publication writes
/// library notes. Drafts and retrieved text stay task-private and survive an explicit Retry.
extension AgentCoordinator {
    func startNoteSeries(
        taskID: UUID,
        request: String,
        selected: [UUID],
        changed: @escaping @Sendable () async -> Void
    ) async throws {
        guard inference is any NoteSeriesInferring || inference is any EvidenceSeriesInferring
        else { throw NoteSeries.Failure.needsChatGPT }
        let run = try await store.beginNoteSeries(
            taskID: taskID,
            request: request,
            provider: inference.capabilities.identifier,
            style: writingStyles().style(for: request),
            selected: selected
        )
        try await continueNoteSeries(run, changed: changed)
        if !run.published {
            await filePublishedSeries(run, changed: changed)
        }
    }

    func continueNoteSeries(
        _ run: NoteSeries.Run,
        changed: @escaping @Sendable () async -> Void
    ) async throws {
        guard !run.published else { return }
        guard run.provider == inference.capabilities.identifier
        else { throw NoteSeries.Failure.providerChanged }
        if let model = inference as? any EvidenceSeriesInferring,
           try await usesEvidenceSeries(run)
        {
            try await continueEvidenceSeries(run, model: model, changed: changed)
            return
        }
        guard let model = inference as? any NoteSeriesInferring,
              let researcher = inference as? any WebResearching
        else { throw NoteSeries.Failure.needsChatGPT }
        let brief = try await seriesBrief(run, model: model)
        try await seriesProgress(
            run,
            "Searching the PKMS for existing coverage",
            owner: .curator,
            changed: changed
        )
        let library = try await seriesLibrary(run, brief: brief)
        try await store.validateSeriesLibrary(library.value)
        let originalOutline = try await seriesOutline(
            run,
            model: model,
            brief: brief,
            library: library
        )
        let newCount = originalOutline.value.entries.filter { $0.existingNoteID == nil }.count
        try await seriesProgress(
            run,
            "Reusing \(originalOutline.value.entries.count - newCount) existing notes, researching \(newCount) gaps",
            owner: .researcher,
            changed: changed
        )
        let researched = try await seriesDossier(
            run,
            brief: brief,
            outline: originalOutline,
            library: library,
            researcher: researcher,
            extractor: model,
            changed: changed
        )
        let prepared = try await prepareSeriesScopes(
            run,
            original: originalOutline,
            dossier: researched,
            model: model,
            changed: changed
        )
        let outline = prepared.outline
        let dossier = prepared.dossier
        var drafts: [UUID: NoteSeries.Draft] = [:]
        var resumedRepair = false
        var parents = [outline.id, dossier.id]
        for entry in outline.value.entries where entry.existingNoteID == nil {
            try await seriesProgress(
                run,
                "Writing \(drafts.count + 1) of \(newCount) notes: \(entry.title)",
                owner: .writer,
                changed: changed
            )
            let draft = try await resumableSeriesDraft(
                run,
                entry: entry,
                dossier: dossier,
                model: model
            )
            drafts[entry.id] = draft.artifact.value
            parents.append(draft.artifact.id)
            resumedRepair = resumedRepair || draft.resumedRepair
        }
        try await seriesProgress(
            run,
            "Checking distinctness and connecting related notes",
            owner: .curator,
            changed: changed
        )
        try await reviewSeries(
            run,
            outline: outline,
            dossier: dossier,
            library: library,
            drafts: drafts,
            parents: parents,
            model: model,
            alreadyRepaired: resumedRepair,
            scope: prepared.bundle,
            changed: changed
        )
        try await seriesProgress(
            run,
            "Publishing the linked note series",
            owner: .writer,
            changed: changed
        )
        _ = try await store.publishNoteSeries(run)
        await changed()
    }

    private struct SeriesReviewContext {
        let run: NoteSeries.Run
        let outline: NoteSeries.Artifact<NoteSeries.Outline>
        let dossier: NoteSeries.Artifact<NoteSeries.Dossier>
        let library: NoteSeries.Artifact<[NoteSeries.Source]>
        let model: any NoteSeriesInferring
        let scope: NoteSeries.ScopeBundle?
        let changed: @Sendable () async -> Void
    }

    private struct SeriesReviewState {
        var outline: NoteSeries.Artifact<NoteSeries.Outline>
        var scope: NoteSeries.ScopeBundle?
        var dossier: NoteSeries.Artifact<NoteSeries.Dossier>
        var drafts: [UUID: NoteSeries.Draft]
        var parents: [UUID]
        var candidate: NoteSeries.Artifact<NoteSeries.Curation>?
        var requirements: NoteSeries.Artifact<NoteSeries.Curation>?
        var feedback: NoteSeries.CurationFeedback?
        var calls = 0
        var repaired = false
        var needsReview = false

        var canRepair: Bool {
            calls < 2 && !repaired
        }
    }

    /// One repair pass per resume, with at most two Curator calls and the shared run budget.
    private func reviewSeries(
        _ run: NoteSeries.Run,
        outline: NoteSeries.Artifact<NoteSeries.Outline>,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        library: NoteSeries.Artifact<[NoteSeries.Source]>,
        drafts: [UUID: NoteSeries.Draft],
        parents: [UUID],
        model: any NoteSeriesInferring,
        alreadyRepaired: Bool,
        scope: NoteSeries.ScopeBundle?,
        changed: @escaping @Sendable () async -> Void
    ) async throws {
        if try await store.seriesArtifact(
            run,
            kind: .reviewedSeries,
            as: NoteSeries.ReviewedSeries.self
        ) != nil {
            return
        }
        let context = SeriesReviewContext(
            run: run,
            outline: outline,
            dossier: dossier,
            library: library,
            model: model,
            scope: scope,
            changed: changed
        )
        var state = try await loadSeriesReview(context, drafts: drafts, parents: parents)
        state.repaired = alreadyRepaired
        while true {
            try Task.checkCancellation()
            if state.needsReview {
                try await generateSeriesReview(context, state: &state)
            }
            if try await acceptOrRepairSeriesReview(context, state: &state) {
                return
            }
        }
    }

    private func loadSeriesReview(
        _ context: SeriesReviewContext,
        drafts: [UUID: NoteSeries.Draft],
        parents: [UUID]
    ) async throws -> SeriesReviewState {
        var state = SeriesReviewState(
            outline: context.outline,
            scope: context.scope,
            dossier: context.dossier,
            drafts: drafts,
            parents: parents
        )
        state.candidate = try await store.latestSeriesArtifact(
            context.run,
            kind: .curationCandidate,
            as: NoteSeries.Curation.self
        )
        if state.candidate == nil {
            state.candidate = try await store.seriesArtifact(
                context.run,
                kind: .curation,
                as: NoteSeries.Curation.self
            )
        }
        state.requirements = try await store.seriesArtifact(
            context.run,
            kind: .curationRequirements,
            as: NoteSeries.Curation.self
        )
        state.needsReview = true
        if let candidate = state.candidate {
            state.needsReview = try await !store.seriesArtifactIncludesParents(
                context.run,
                artifactID: candidate.id,
                parents: parents
            )
            if state.needsReview {
                state.feedback = .init(
                    previous: candidate.value,
                    issues: [],
                    requiredLinks: state.requirements?.value.links ?? candidate
                        .value.links
                )
            }
        }
        return state
    }

    private func generateSeriesReview(
        _ context: SeriesReviewContext,
        state: inout SeriesReviewState
    ) async throws {
        while true {
            guard state.calls < 2 else { throw NoteSeries.Failure.invalidLinks }
            let value: NoteSeries.Curation
            if state.drafts.isEmpty {
                value = .init(distinct: true, links: [])
            } else {
                try await store.reserveSeriesCall(context.run)
                state.calls += 1
                do {
                    value = try await context.model.curateSeries(
                        state.outline.value,
                        drafts: state.drafts,
                        library: context.library.value,
                        feedback: state.feedback
                    )
                } catch NoteSeries.Failure.invalidLinks {
                    guard state.canRepair else { throw NoteSeries.Failure.invalidLinks }
                    state.repaired = true
                    state.feedback = .init(
                        previous: nil,
                        issues: [.init(
                            kind: .invalidTarget,
                            message: "The last response used an unknown note handle."
                        )],
                        requiredLinks: state.requirements?.value.links ?? []
                    )
                    continue
                }
            }
            state.candidate = try await store.checkpointSeries(
                context.run,
                kind: .curationCandidate,
                slot: UUID().uuidString,
                value: value,
                parents: state.parents
            )
            state.needsReview = false
            return
        }
    }

    private func acceptOrRepairSeriesReview(
        _ context: SeriesReviewContext,
        state: inout SeriesReviewState
    ) async throws -> Bool {
        guard let candidate = state.candidate else { throw NoteSeries.Failure.invalidArtifact }
        if state.requirements == nil {
            state.requirements = try await store.checkpointSeries(
                context.run,
                kind: .curationRequirements,
                value: candidate.value,
                parents: [candidate.id]
            )
        }
        guard let requirements = state.requirements
        else { throw NoteSeries.Failure.invalidArtifact }
        var issues = candidate.value.issues(
            outline: state.outline.value,
            drafts: state.drafts,
            library: context.library.value
        )
        do {
            try candidate.value.validatePreservedLinks(
                from: requirements.value,
                outline: state.outline.value,
                drafts: state.drafts,
                library: context.library.value
            )
        } catch let NoteSeries.Failure.curation(issue) {
            issues.append(issue)
        }
        guard let issue = issues.first else {
            let reviewed = NoteSeries.ReviewedSeries(
                drafts: state.drafts,
                curation: candidate.value,
                requirements: requirements.value,
                evidence: state.dossier.value,
                scope: state.scope
            )
            guard let original = try await store.seriesArtifact(
                context.run,
                kind: .outline,
                as: NoteSeries.Outline.self
            ),
                let evidence = try await store.seriesArtifact(
                    context.run,
                    kind: .dossier,
                    as: NoteSeries.Dossier.self
                )
            else { throw NoteSeries.Failure.invalidArtifact }
            try reviewed.validate(outline: original.value, dossier: evidence.value)
            try await store.checkpointSeries(
                context.run,
                kind: .reviewedSeries,
                value: reviewed,
                parents: state.parents + [
                    candidate.id,
                    requirements.id,
                ]
            )
            return true
        }
        try await exhaustResearchedOverlaps(context, state: state, issues: issues)
        guard state.canRepair else { throw NoteSeries.Failure.curation(issue) }
        state.feedback = .init(
            previous: candidate.value,
            issues: issues,
            requiredLinks: requirements.value.links
        )
        // Legacy Boolean-only rejections need a diagnosis before choosing drafts to revise.
        if !issues.allSatisfy({ $0.kind == .missingOverlapDetails }) {
            state.repaired = true
            try await reviseOverlappingSeriesDrafts(context, state: &state, issues: issues)
        }
        try await seriesProgress(
            context.run,
            "Repairing and rechecking the final review",
            owner: .curator,
            changed: context.changed
        )
        state.needsReview = true
        return false
    }

    private func reviseOverlappingSeriesDrafts(
        _ context: SeriesReviewContext,
        state: inout SeriesReviewState,
        issues: [NoteSeries.CurationIssue]
    ) async throws {
        guard issues.contains(where: { $0.kind == .overlap }),
              !issues.contains(where: { $0.kind == .invalidOverlap }),
              let candidate = state.candidate else { return }
        let overlaps = candidate.value.overlapsToRepair(
            outline: state.outline.value,
            drafts: state.drafts
        )
        for (index, planned) in state.outline.value.entries.enumerated() {
            var entry = planned
            guard let overlap = overlaps.first(where: { $0.from == entry.id }),
                  let draft = state.drafts[entry.id] else { continue }
            let comparison = try seriesComparisonNote(
                overlap.to,
                context: context,
                outline: state.outline.value,
                drafts: state.drafts
            )
            try await seriesProgress(
                context.run,
                "Revising overlapping note: \(entry.title)",
                owner: .writer,
                changed: context.changed
            )
            let repair = NoteSeries.DraftFeedback(
                draft: draft,
                structurallyValid: true,
                supported: draft.paragraphs.map { _ in true },
                distinctness: .init(
                    reason: overlap.reason,
                    comparison: comparison
                )
            )
            let recovered = try await recoverSeriesTopic(
                context,
                state: &state,
                entry: entry,
                feedback: repair
            )
            state.dossier = recovered
            guard let current = state.outline.value.entries.first(where: { $0.id == entry.id })
            else { throw NoteSeries.Failure.invalidArtifact }
            entry = current
            if !state.parents.contains(state.dossier.id) {
                state.parents.append(state.dossier.id)
            }
            let revision: NoteSeries.Artifact<NoteSeries.Draft>
            do {
                revision = try await seriesDraft(
                    context.run,
                    entry: entry,
                    dossier: state.dossier,
                    model: context.model,
                    repairFeedback: repair,
                    repairParents: state.parents + [candidate.id]
                )
            } catch let NoteSeries.Failure.draft(_, issue) {
                try await failSeriesRevision(
                    context,
                    state: state,
                    entry: entry,
                    index: index,
                    issue: issue
                )

            } catch NoteSeries.Failure.invalidDraft {
                throw NoteSeries.Failure.curation(.init(
                    kind: .unsupportedRevision,
                    message: "The revision for note \(index + 1) failed its citation or source-support checks."
                ))
            }
            state.drafts[entry.id] = revision.value
            let retained = try await retainScopedSeriesDraft(
                context.run,
                entry: entry,
                draft: revision,
                dossier: state.dossier
            )
            state.parents.append(retained.id)
        }
    }

    private func failSeriesRevision(
        _ context: SeriesReviewContext, state: SeriesReviewState,
        entry: NoteSeries.Entry, index: Int, issue: NoteSeries.DraftIssue
    ) async throws -> Never {
        if try await store.seriesArtifact(
            context.run,
            kind: .topicSearch,
            slot: entry.id.uuidString,
            as: WebResearchResult.self
        ) !=
            nil
        {
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: String(issue.message.prefix(700)),
                parents: state.parents
            )
        }
        throw NoteSeries.Failure.curation(.init(
            kind: .unsupportedRevision,
            message: "The revision for note \(index + 1) (\(entry.title)): \(issue.message)"
        ))
    }

    private func exhaustResearchedOverlaps(
        _ context: SeriesReviewContext,
        state: SeriesReviewState,
        issues: [NoteSeries.CurationIssue]
    ) async throws {
        guard state.repaired, issues.contains(where: { $0.kind == .overlap }),
              !issues.contains(where: { $0.kind == .invalidOverlap }),
              let candidate = state.candidate else { return }
        for overlap in candidate.value.overlapsToRepair(
            outline: state.outline.value,
            drafts: state.drafts
        ) {
            guard let entry = state.outline.value.entries.first(where: { $0.id == overlap.from }),
                  try await store.seriesArtifact(
                      context.run,
                      kind: .topicSearch,
                      slot: entry.id.uuidString,
                      as: WebResearchResult.self
                  ) != nil else { continue }
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: String(overlap.reason.prefix(700)),
                parents: state.parents + [candidate.id]
            )
        }
    }

    private func recoverSeriesTopic(
        _ context: SeriesReviewContext,
        state: inout SeriesReviewState,
        entry: NoteSeries.Entry,
        feedback: NoteSeries.DraftFeedback
    ) async throws -> NoteSeries.Artifact<NoteSeries.Dossier> {
        do {
            return try await recoverSeriesTopicEvidence(
                context,
                state: state,
                entry: entry,
                feedback: feedback
            )
        } catch let NoteSeries.Failure.insufficientTopicEvidence(_, reason) {
            try await seriesProgress(
                context.run,
                "Checking a supported scope for: \(entry.title)",
                owner: .curator,
                changed: context.changed
            )
            guard let original = try await store.seriesArtifact(
                context.run,
                kind: .outline,
                as: NoteSeries.Outline.self
            )
            else { throw NoteSeries.Failure.invalidArtifact }
            let prepared = PreparedSeriesScope(
                outline: state.outline,
                dossier: state.dossier,
                bundle: state.scope
            )
            let narrowed = try await narrowSeriesScope(
                context.run,
                original: original,
                prepared: prepared,
                entry: entry,
                feedback: feedback,
                failure: reason,
                model: context.model
            )
            state.outline = narrowed.outline
            state.scope = narrowed.bundle
            return narrowed.dossier
        }
    }

    private func recoverSeriesTopicEvidence(
        _ context: SeriesReviewContext,
        state: SeriesReviewState,
        entry: NoteSeries.Entry,
        feedback: NoteSeries.DraftFeedback
    ) async throws -> NoteSeries.Artifact<NoteSeries.Dossier> {
        if let exhausted = try await store.seriesArtifact(
            context.run,
            kind: .topicExhausted,
            slot: entry.id.uuidString,
            as: String.self
        ) {
            throw NoteSeries.Failure.insufficientTopicEvidence(
                note: entry.title,
                reason: exhausted.value
            )
        }
        if state.dossier.value.topicRecords?[entry.id] != nil {
            return state.dossier
        }
        if try await store.seriesArtifact(
            context.run,
            kind: .topicRepair,
            slot: entry.id.uuidString,
            as: NoteSeries.DraftFeedback.self
        ) == nil {
            try await store.checkpointSeries(
                context.run,
                kind: .topicRepair,
                slot: entry.id.uuidString,
                value: feedback,
                parents: state.parents
            )
        }
        try await seriesProgress(
            context.run,
            "Checking evidence for: \(entry.title)",
            owner: .researcher,
            changed: context.changed
        )
        let plan = try await assessSeriesTopic(
            context,
            entry: entry,
            dossier: state.dossier,
            feedback: feedback,
            kind: .topicPlan
        )
        var dossier = state.dossier.value
        var parents = state.parents + [plan.id]
        var assessment = plan
        if !plan.value.sufficient {
            let enriched = try await researchSeriesTopic(
                context,
                entry: entry,
                dossier: state.dossier,
                plan: plan
            )
            dossier = enriched.value
            parents.append(enriched.id)
            assessment = try await assessSeriesTopic(
                context,
                entry: entry,
                dossier: enriched,
                feedback: feedback,
                kind: .topicAssessment
            )
            parents.append(assessment.id)
        }
        guard assessment.value.sufficient else {
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: assessment.value.reason,
                parents: parents
            )
        }
        var selected = dossier.topicRecords ?? [:]
        selected[entry.id] = assessment.value.records
        dossier.topicRecords = selected
        try dossier.validateExtending(state.dossier.value)
        return try await store.checkpointSeries(
            context.run,
            kind: .topicDossier,
            slot: entry.id.uuidString,
            value: dossier,
            parents: parents
        )
    }

    private func assessSeriesTopic(
        _ context: SeriesReviewContext,
        entry: NoteSeries.Entry,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        feedback: NoteSeries.DraftFeedback,
        kind: NoteSeries.ArtifactKind
    ) async throws -> NoteSeries.Artifact<NoteSeries.TopicAssessment> {
        if let cached = try await store.seriesArtifact(
            context.run,
            kind: kind,
            slot: entry.id.uuidString,
            as: NoteSeries.TopicAssessment.self
        ) {
            try cached.value.validate(dossier: dossier.value)
            return cached
        }
        try await store.reserveSeriesCall(context.run)
        let value = try await context.model.assessSeriesTopic(
            entry,
            dossier: dossier.value,
            feedback: feedback
        )
        try value.validate(dossier: dossier.value)
        return try await store.checkpointSeries(
            context.run,
            kind: kind,
            slot: entry.id.uuidString,
            value: value,
            parents: [dossier.id]
        )
    }

    private func researchSeriesTopic(
        _ context: SeriesReviewContext,
        entry: NoteSeries.Entry,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        plan: NoteSeries.Artifact<NoteSeries.TopicAssessment>
    ) async throws -> NoteSeries.Artifact<NoteSeries.Dossier> {
        // This intermediate snapshot also makes interruption after extraction resumable.
        let slot = entry.id.uuidString + ".research"
        if let cached = try await store.seriesArtifact(
            context.run,
            kind: .topicResearch,
            slot: slot,
            as: NoteSeries.Dossier.self
        ) {
            try cached.value.validateExtending(dossier.value)
            return cached
        }
        guard let brief = try await store.seriesArtifact(
            context.run,
            kind: .brief,
            as: NoteSeries.Brief.self
        )
        else { throw NoteSeries.Failure.invalidArtifact }
        guard brief.value.mode != .libraryOnly, !plan.value.query.isEmpty else {
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: "Saved evidence does not establish this note's distinct coverage, and its source policy or lack of a safe public query prevents targeted web research.",
                parents: [plan.id]
            )
        }
        let web = try await seriesTopicSearch(context, entry: entry, plan: plan)
        let pages = try await seriesWebSources(
            context.run,
            web: web,
            limit: NoteSeries.topicSourceLimit,
            attemptPrefix: entry.id.uuidString + ".",
            changed: context.changed
        )
        var sources = dossier.value.sources
        var facts = dossier.value.facts
        var added: [String] = []
        var parents = [dossier.id, plan.id, web.id]
        for page in pages.sources {
            if !sources.contains(where: { $0.id == page.value.id }) {
                sources.append(page.value)
            }
            let extracted = try await seriesFacts(
                context.run,
                source: page.value,
                fields: entry.fields,
                extractor: context.model,
                parents: [page.id, plan.id],
                slotPrefix: entry.id.uuidString + ".",
                request: "\(context.run.request)\nFocus only on this note: \(entry.title). \(entry.purpose)"
            )
            parents += [page.id, extracted.id]
            for (offset, record) in extracted.value.prefix(NoteSeries.factsPerSource).enumerated() {
                let id = "T\(entry.id.uuidString)S\(page.value.id.uuidString)R\(offset + 1)"
                facts.append(.init(
                    id: id,
                    sourceID: page.value.id,
                    field: record.field,
                    value: record.value,
                    quote: record.quote
                ))
                added.append(id)
            }
        }
        guard !added.isEmpty else {
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: "The targeted pages provided no additional verified quotations for this note. Search summaries were not used as evidence.",
                parents: parents
            )
        }
        var selected = dossier.value.topicRecords ?? [:]
        selected[entry.id] = added
        let enriched = NoteSeries.Dossier(
            sources: sources,
            facts: facts,
            unavailableURLs: dossier.value.unavailableURLs + pages
                .unavailable,
            topicRecords: selected,
            recoverySearchCount: (dossier.value
                .recoverySearchCount ?? 0) + web.value.searchCount
        )
        try enriched.validateExtending(dossier.value)
        return try await store.checkpointSeries(
            context.run,
            kind: .topicResearch,
            slot: slot,
            value: enriched,
            parents: parents
        )
    }

    private func seriesTopicSearch(
        _ context: SeriesReviewContext,
        entry: NoteSeries.Entry,
        plan: NoteSeries.Artifact<NoteSeries.TopicAssessment>
    ) async throws -> NoteSeries.Artifact<WebResearchResult> {
        if let cached = try await store.seriesArtifact(
            context.run,
            kind: .topicSearch,
            slot: entry.id.uuidString,
            as: WebResearchResult.self
        ) {
            return cached
        }
        guard try await store.seriesArtifact(
            context.run,
            kind: .topicSearchStarted,
            slot: entry.id.uuidString,
            as: String.self
        ) == nil
        else {
            try await exhaustSeriesTopic(
                context.run,
                entry: entry,
                reason: "The single targeted search was interrupted before a usable result was saved. Retry will not start another search for this topic.",
                parents: [plan.id]
            )
        }
        guard let researcher = context.model as? any WebResearching
        else { throw NoteSeries.Failure.needsChatGPT }
        try await store.reserveSeriesCall(context.run)
        try await store.checkpointSeries(
            context.run,
            kind: .topicSearchStarted,
            slot: entry.id.uuidString,
            value: plan.value.query,
            parents: [plan.id]
        )
        let store = store
        // Only the public keyword query reaches search. No draft, PKMS text or critique is sent.
        let result = try await researcher.researchWeb(
            question: "Research \(plan.value.query). Find up to \(NoteSeries.topicSourceLimit) authoritative substantive pages with specific evidence for this public topic, not generic overviews.",
            sources: []
        ) { text in
            try? await store.updateConversationTask(
                context.run.taskID,
                progress: text,
                owner: .researcher
            )
            await context.changed()
        }
        _ = try result.markdown()
        return try await store.checkpointSeries(
            context.run,
            kind: .topicSearch,
            slot: entry.id.uuidString,
            value: result,
            parents: [plan.id]
        )
    }

    private func exhaustSeriesTopic(
        _ run: NoteSeries.Run, entry: NoteSeries.Entry, reason: String, parents: [UUID]
    ) async throws -> Never {
        let saved = try await store.seriesArtifact(
            run,
            kind: .topicExhausted,
            slot: entry.id.uuidString,
            as: String.self
        )
        if saved == nil {
            try await store.checkpointSeries(
                run,
                kind: .topicExhausted,
                slot: entry.id.uuidString,
                value: reason,
                parents: parents
            )
        }
        throw NoteSeries.Failure.insufficientTopicEvidence(
            note: entry.title,
            reason: saved?.value ?? reason
        )
    }

    private func seriesComparisonNote(
        _ id: UUID,
        context: SeriesReviewContext,
        outline: NoteSeries.Outline,
        drafts: [UUID: NoteSeries.Draft]
    ) throws -> NoteSeries.ReviewNote {
        if let draft = drafts[id],
           let entry = outline.entries.first(where: { $0.id == id })
        {
            return .init(
                id: id,
                title: entry.title,
                text: draft.paragraphs.map(\.text).joined(separator: "\n")
            )
        }
        guard let source = context.library.value.first(where: { $0.noteID == id }) else {
            throw NoteSeries.Failure.invalidArtifact
        }
        return .init(id: id, title: source.title, text: source.text)
    }

    private func seriesBrief(
        _ run: NoteSeries.Run,
        model: any NoteSeriesInferring
    ) async throws -> NoteSeries.Artifact<NoteSeries.Brief> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .brief,
            as: NoteSeries.Brief.self
        ) {
            return cached
        }
        try await store.reserveSeriesCall(run)
        let brief = try await model.seriesBrief(request: run.request)
        guard NoteSeries.plain(brief.query, limit: 200), brief.libraryQueries.count <= 6,
              brief.libraryQueries.allSatisfy({ NoteSeries.plain($0, limit: 200) })
        else { throw NoteSeries.Failure.invalidOutline }
        return try await store.checkpointSeries(run, kind: .brief, value: brief, parents: [])
    }

    func seriesLibrary(
        _ run: NoteSeries.Run,
        brief: NoteSeries.Artifact<NoteSeries.Brief>
    ) async throws -> NoteSeries.Artifact<[NoteSeries.Source]> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .library,
            as: [NoteSeries.Source].self
        ) {
            return cached
        }
        let queries = brief.value.libraryQueries.isEmpty ? [brief.value.query] : brief.value
            .libraryQueries
        var results: [[SearchMatch]] = []
        for query in queries {
            try await results.append(search.search(query, limit: 80).matches)
        }
        // Interleave query ranks so one broad phrase cannot crowd every focused concept out.
        let matches = (0 ..< 80)
            .flatMap { rank in results.compactMap { $0.indices.contains(rank) ? $0[rank] : nil } }
        var seen = Set<UUID>()
        let ids = (run.selected + matches.map(\.passage.documentID))
            .filter { seen.insert($0).inserted }.prefix(NoteSeries.maximumLibrarySources)
        var sources: [NoteSeries.Source] = []
        for id in ids {
            guard let note = try await store.document(id: id) else { continue }
            let text = String(note.markdown.prefix(NoteSeries.sourceCharacters))
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            sources.append(NoteSeries.Source(
                id: note.id,
                title: note.name,
                text: text,
                url: nil,
                noteID: note.id,
                revision: note.revision,
                contentHash: ContentHash.of(text),
                retrievedAt: Date(),
                limitations: note.markdown.count > text
                    .count ?
                    ["Only the first \(NoteSeries.sourceCharacters) characters were read."] :
                    []
            ))
        }
        return try await store.checkpointSeries(
            run,
            kind: .library,
            value: sources,
            parents: [brief.id]
        )
    }

    private func seriesOutline(
        _ run: NoteSeries.Run,
        model: any NoteSeriesInferring,
        brief: NoteSeries.Artifact<NoteSeries.Brief>,
        library: NoteSeries.Artifact<[NoteSeries.Source]>
    ) async throws -> NoteSeries.Artifact<NoteSeries.Outline> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .outline,
            as: NoteSeries.Outline.self
        ) {
            return cached
        }
        try await store.reserveSeriesCall(run)
        let outline = try await model.seriesOutline(
            request: run.request,
            brief: brief.value,
            library: library.value,
            style: run.style
        )
        try outline.validate(library: library.value, brief: brief.value)
        try await store.reserveSeriesCall(run)
        guard try await model.reviewSeriesOutline(
            outline,
            request: run.request,
            library: library.value
        )
        else { throw NoteSeries.Failure.invalidOutline }
        return try await store.checkpointSeries(
            run,
            kind: .outline,
            value: outline,
            parents: [brief.id, library.id]
        )
    }

    private func seriesDossier(
        _ run: NoteSeries.Run,
        brief: NoteSeries.Artifact<NoteSeries.Brief>,
        outline: NoteSeries.Artifact<NoteSeries.Outline>,
        library: NoteSeries.Artifact<[NoteSeries.Source]>,
        researcher: any WebResearching,
        extractor: any NoteSeriesInferring,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> NoteSeries.Artifact<NoteSeries.Dossier> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .dossier,
            as: NoteSeries.Dossier.self
        ) {
            if let recovered = try await store.latestSeriesArtifact(
                run,
                kind: .topicDossier,
                as: NoteSeries.Dossier.self
            ) {
                try recovered.value.validateExtending(cached.value)
                return recovered
            }
            return cached
        }
        let newEntries = outline.value.entries.filter { $0.existingNoteID == nil }
        var sources = library.value
        var unavailable: [URL] = []
        var parents = [outline.id, library.id]
        if brief.value.mode != .libraryOnly, !newEntries.isEmpty {
            let sourceLimit = NoteSeries.webSourceLimit(newNotes: newEntries.count)
            let web = try await seriesWebSearch(
                run,
                brief: brief,
                researcher: researcher,
                parent: outline.id,
                sourceLimit: sourceLimit,
                changed: changed
            )
            let pages = try await seriesWebSources(
                run,
                web: web,
                limit: sourceLimit,
                changed: changed
            )
            guard !pages.sources.isEmpty else { throw NoteSeries.Failure.noWebSources }
            sources += pages.sources.map(\.value)
            parents += pages.sources.map(\.id)
            unavailable = pages.unavailable
        }
        var records: [NoteSeries.Fact] = []
        if !newEntries.isEmpty {
            let fields = Array(Set(newEntries.flatMap(\.fields))).sorted()
            let sourceParents = parents
            for (index, source) in sources.enumerated() {
                try await seriesProgress(
                    run,
                    "Extracting evidence \(index + 1) of \(sources.count)",
                    owner: .researcher,
                    changed: changed
                )
                let facts = try await seriesFacts(
                    run,
                    source: source,
                    fields: fields,
                    extractor: extractor,
                    parents: sourceParents
                )
                parents.append(facts.id)
                records += facts.value.prefix(NoteSeries.factsPerSource).enumerated()
                    .map { offset, fact in
                        NoteSeries.Fact(
                            id: "E\(index + 1)R\(offset + 1)",
                            sourceID: source.id,
                            field: fact.field,
                            value: fact.value,
                            quote: fact.quote
                        )
                    }
            }
        }
        let dossier = NoteSeries.Dossier(
            sources: sources,
            facts: records,
            unavailableURLs: unavailable
        )
        try dossier.validate()
        return try await store.checkpointSeries(
            run,
            kind: .dossier,
            value: dossier,
            parents: parents
        )
    }

    private func seriesWebSearch(
        _ run: NoteSeries.Run,
        brief: NoteSeries.Artifact<NoteSeries.Brief>,
        researcher: any WebResearching,
        parent: UUID,
        sourceLimit: Int,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> NoteSeries.Artifact<WebResearchResult> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .webSearch,
            as: WebResearchResult.self
        ) {
            return cached
        }
        try await store.reserveSeriesCall(run)
        let store = store
        // Only the public query derived from the user's request reaches hosted search. PKMS
        // excerpts and outline text never enter an external search request.
        let web = try await researcher.researchWeb(
            question: "Research \(brief.value.query). Map the topic's distinct principal concepts and find diverse authoritative sources covering them, not just general overviews. Cite up to \(sourceLimit) distinct substantive source pages, as coverage requires; prefer complementary primary sources. A smaller topic needs fewer sources.",
            sources: []
        ) { text in
            try? await store.updateConversationTask(run.taskID, progress: text, owner: .researcher)
            await changed()
        }
        _ = try web.markdown()
        return try await store.checkpointSeries(
            run,
            kind: .webSearch,
            value: web,
            parents: [parent]
        )
    }

    private func seriesWebSources(
        _ run: NoteSeries.Run,
        web: NoteSeries.Artifact<WebResearchResult>,
        limit: Int,
        attemptPrefix: String? = nil,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> (sources: [NoteSeries.Artifact<NoteSeries.Source>], unavailable: [URL]) {
        var seen = Set<URL>()
        let urls = web.value.passages.flatMap(\.citations).map(\.url)
            .filter { seen.insert($0).inserted }.prefix(limit)
        var sources: [NoteSeries.Artifact<NoteSeries.Source>] = []
        var unavailable: [URL] = []
        var hashes = Set<String>()
        for url in urls {
            try await seriesProgress(
                run,
                "Reading web source \(sources.count + unavailable.count + 1) of \(urls.count)",
                owner: .researcher,
                changed: changed
            )
            do {
                let source = try await seriesWebSource(
                    run,
                    url: url,
                    parent: web.id,
                    attemptPrefix: attemptPrefix
                )
                if hashes.insert(source.value.contentHash).inserted {
                    sources.append(source)
                }
            } catch is CancellationError { throw CancellationError() }
            catch let error as NoteSeries.Failure { throw error }
            catch { unavailable.append(url) }
        }
        return (sources, unavailable)
    }

    func seriesWebSource(
        _ run: NoteSeries.Run, url: URL, parent: UUID, attemptPrefix: String?
    ) async throws -> NoteSeries.Artifact<NoteSeries.Source> {
        let slot = ContentHash.of(url.absoluteString)
        if let cached = try await store.seriesArtifact(
            run,
            kind: .webSource,
            slot: slot,
            as: NoteSeries.Source.self
        ) {
            return cached
        }
        if let prefix = attemptPrefix {
            guard try await store.seriesArtifact(
                run,
                kind: .topicSourceAttempt,
                slot: prefix + slot,
                as: URL.self
            ) == nil
            else { throw CaptureError.downloadFailed }
            try await store.checkpointSeries(
                run,
                kind: .topicSourceAttempt,
                slot: prefix + slot,
                value: url,
                parents: [parent]
            )
        }
        let downloaded = try await researchFetcher.fetch(url)
        let value = try await Self.readSeriesSource(downloaded)
        return try await store.checkpointSeries(
            run,
            kind: .webSource,
            slot: slot,
            value: value,
            parents: [parent]
        )
    }

    private static func readSeriesSource(_ downloaded: DownloadedContent) async throws -> NoteSeries
        .Source
    {
        guard WebResearchResult.safeURL(downloaded.url),
              downloaded.data.count <= 8_000_000 else { throw CaptureError.tooLarge }
        let document: CapturedDocument
        if downloaded.mime == "application/pdf" {
            document = try await PDFExtractor().extract(
                downloaded.data,
                name: "Research source",
                url: downloaded.url
            )
        } else if ["text/plain", "text/markdown", "text/x-markdown"].contains(downloaded.mime) {
            let text = String(decoding: downloaded.data, as: UTF8.self)
            document = CapturedDocument(
                title: downloaded.url.host ?? "Research source",
                sourceURL: downloaded.url,
                coverage: .document,
                blocks: [ContentBlock(kind: .paragraph, markdown: text, text: text)]
            )
        } else {
            guard downloaded.mime == nil || downloaded.mime?.contains("html") == true
            else { throw CaptureError.unsupported }
            let html = String(data: downloaded.data, encoding: .utf8) ?? String(
                data: downloaded.data,
                encoding: .isoLatin1
            ) ?? ""
            document = try await HTMLExtractor.extract(
                HTMLExtractor.analyze(html, url: downloaded.url),
                url: downloaded.url
            )
        }
        let full = document.evidenceBlocks.map(\.text).joined(separator: "\n\n")
        let text = String(full.prefix(NoteSeries.sourceCharacters))
        guard text.count >= 80 else { throw CaptureError.noContent }
        return NoteSeries.Source(
            id: UUID(),
            title: String(document.title.prefix(1000)),
            text: text,
            url: downloaded.url,
            noteID: nil,
            revision: nil,
            contentHash: ContentHash.of(text),
            retrievedAt: Date(),
            limitations: ["Captured coverage: \(document.coverage.rawValue)."] + document
                .warnings +
                (full.count > text
                    .count ?
                    ["Only the first \(NoteSeries.sourceCharacters) characters were read."] :
                    [])
        )
    }

    func seriesFacts(
        _ run: NoteSeries.Run,
        source: NoteSeries.Source,
        fields: [String],
        extractor: any SeriesEvidenceInferring,
        parents: [UUID],
        slotPrefix: String = "",
        request: String? = nil
    ) async throws -> NoteSeries.Artifact<[FactRecord]> {
        if let cached = try await store.seriesArtifact(
            run,
            kind: .facts,
            slot: slotPrefix + source.id.uuidString,
            as: [FactRecord].self
        ) {
            return cached
        }
        try await store.reserveSeriesCall(run)
        let facts = try await extractor.extractSeriesFacts(ExtractionChunk(
            title: source.title,
            text: source.text,
            fields: fields,
            request: request ?? run.request,
            part: 1,
            parts: 1
        ))
        let checked = FactRecord.verified(
            facts,
            in: source.text,
            fields: fields,
            sourceID: source.id
        )
        return try await store.checkpointSeries(
            run,
            kind: .facts,
            slot: slotPrefix + source.id.uuidString,
            value: Array(checked.prefix(NoteSeries.factsPerSource)),
            parents: parents
        )
    }

    private func resumableSeriesDraft(
        _ run: NoteSeries.Run, entry: NoteSeries.Entry,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        model: any NoteSeriesInferring
    ) async throws -> (artifact: NoteSeries.Artifact<NoteSeries.Draft>, resumedRepair: Bool) {
        if dossier.value.scopeRecords?[entry.id] != nil {
            return try await resumableScopedSeriesDraft(
                run,
                entry: entry,
                dossier: dossier,
                model: model
            )
        }
        if let exhausted = try await store.seriesArtifact(
            run,
            kind: .topicExhausted,
            slot: entry.id.uuidString,
            as: String.self
        ) {
            throw NoteSeries.Failure.insufficientTopicEvidence(
                note: entry.title,
                reason: exhausted.value
            )
        }
        if dossier.value.topicRecords?[entry.id] != nil,
           try await store.latestSeriesArtifact(
               run,
               kind: .draftRevision,
               slotPrefix: entry.id.uuidString + ".",
               as: NoteSeries.Draft.self
           ) == nil
        {
            guard let repair = try await store.seriesArtifact(
                run,
                kind: .topicRepair,
                slot: entry.id.uuidString,
                as: NoteSeries.DraftFeedback.self
            )
            else { throw NoteSeries.Failure.invalidArtifact }
            do {
                let draft = try await seriesDraft(
                    run,
                    entry: entry,
                    dossier: dossier,
                    model: model,
                    repairFeedback: repair.value,
                    repairParents: [dossier.id]
                )
                return (draft, true)
            } catch let NoteSeries.Failure.draft(note, issue) {
                if try await store.seriesArtifact(
                    run,
                    kind: .topicSearch,
                    slot: entry.id.uuidString,
                    as: WebResearchResult.self
                ) != nil {
                    try await exhaustSeriesTopic(
                        run,
                        entry: entry,
                        reason: String(issue.message.prefix(700)),
                        parents: [dossier.id]
                    )
                }
                throw NoteSeries.Failure.draft(note: note, issue: issue)
            }
        }
        return try await (seriesDraft(run, entry: entry, dossier: dossier, model: model), false)
    }

    private func resumableScopedSeriesDraft(
        _ run: NoteSeries.Run, entry: NoteSeries.Entry,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        model: any NoteSeriesInferring
    ) async throws -> (artifact: NoteSeries.Artifact<NoteSeries.Draft>, resumedRepair: Bool) {
        if let cached = try await store.latestSeriesArtifact(
            run,
            kind: .scopeDraft,
            slotPrefix: entry.id.uuidString + ".",
            as: NoteSeries.ScopedDraft.self
        ), cached.value.entry == entry {
            try cached.value.draft.validate(entry: entry, dossier: dossier.value)
            return (.init(id: cached.id, value: cached.value.draft), false)
        }
        guard let feedback = try await store.seriesArtifact(
            run,
            kind: .topicRepair,
            slot: entry.id.uuidString,
            as: NoteSeries.DraftFeedback.self
        )
        else { throw NoteSeries.Failure.invalidArtifact }
        let draft = try await seriesDraft(
            run,
            entry: entry,
            dossier: dossier,
            model: model,
            repairFeedback: feedback.value,
            repairParents: [dossier.id]
        )
        return try await (
            retainScopedSeriesDraft(run, entry: entry, draft: draft, dossier: dossier),
            true
        )
    }

    private func seriesDraft(
        _ run: NoteSeries.Run,
        entry: NoteSeries.Entry,
        dossier: NoteSeries.Artifact<NoteSeries.Dossier>,
        model: any NoteSeriesInferring,
        repairFeedback: NoteSeries.DraftFeedback? = nil,
        repairParents: [UUID] = []
    ) async throws -> NoteSeries.Artifact<NoteSeries.Draft> {
        if repairFeedback == nil {
            if let revision = try await store.latestSeriesArtifact(
                run,
                kind: .draftRevision,
                slotPrefix: entry.id
                    .uuidString + ".",
                as: NoteSeries.Draft.self
            ) {
                return revision
            }
            if let cached = try await store.seriesArtifact(
                run,
                kind: .draft,
                slot: entry.id.uuidString,
                as: NoteSeries.Draft.self
            ) {
                return cached
            }
        }
        guard !dossier.value.facts(for: entry).isEmpty else { throw NoteSeries.Failure.noFacts }
        var feedback = try await seriesDraftFeedback(
            run, entry: entry, dossierID: dossier.id, repairFeedback: repairFeedback,
            repairParents: repairParents
        )
        for _ in 0 ..< 2 {
            try await store.reserveSeriesCall(run)
            let draft = try await model.draftSeriesNote(
                entry,
                request: run.request,
                dossier: dossier.value,
                style: run.style,
                feedback: feedback
            )
            let structuralIssue = draft.validationIssue(entry: entry, dossier: dossier.value)
            let structurallyValid = structuralIssue == nil
            var review: [NoteSeries.ParagraphReview] = []
            if structurallyValid {
                try await store.reserveSeriesCall(run)
                review = try await model.verifySeriesDraft(
                    draft,
                    entry: entry,
                    dossier: dossier.value
                )
            }
            let result = NoteSeries.DraftFeedback(
                draft: draft,
                structurallyValid: structurallyValid,
                supported: draft.paragraphs.enumerated().map { index, paragraph in
                    guard review.indices.contains(index) else { return false }
                    let verdict = review[index]
                    return verdict.verdict == .supported && verdict.isValid(
                        for: paragraph.text,
                        number: index + 1
                    )
                },
                distinctness: repairFeedback?.distinctness,
                issue: structuralIssue ?? draft.verificationIssue(reviews: review),
                reviews: structurallyValid ? review : nil
            )
            feedback = result
            let attempt = try await store.checkpointSeries(
                run,
                kind: .draftFeedback,
                slot: entry.id.uuidString + "." + UUID().uuidString,
                value: result,
                parents: [dossier.id] + repairParents
            )
            guard structurallyValid, result.issue == nil else { continue }
            return try await store.checkpointSeries(
                run,
                kind: repairFeedback == nil ? .draft : .draftRevision,
                slot: entry.id.uuidString + (repairFeedback == nil ? "" : "." + UUID().uuidString),
                value: draft,
                parents: [attempt.id]
            )
        }
        if let issue = feedback?.issue {
            throw NoteSeries.Failure.draft(note: entry.title, issue: issue)
        }
        throw NoteSeries.Failure.invalidDraft
    }

    private func seriesDraftFeedback(
        _ run: NoteSeries.Run,
        entry: NoteSeries.Entry,
        dossierID: UUID,
        repairFeedback: NoteSeries.DraftFeedback?,
        repairParents: [UUID]
    ) async throws -> NoteSeries.DraftFeedback? {
        guard let saved = try await store.latestSeriesArtifact(
            run, kind: .draftFeedback, slotPrefix: entry.id.uuidString + ".",
            as: NoteSeries.DraftFeedback.self
        ), try await store.seriesArtifactIncludesParents(
            run,
            artifactID: saved.id,
            parents: [dossierID] + repairParents
        ), saved.value.distinctness == repairFeedback?.distinctness
        else { return repairFeedback }
        return saved.value
    }

    func seriesProgress(
        _ run: NoteSeries.Run,
        _ text: String,
        owner: AgentRole,
        changed: @escaping @Sendable () async -> Void
    ) async throws {
        try Task.checkCancellation()
        try await store.updateConversationTask(
            run.taskID,
            progress: text,
            owner: owner,
            event: "delegated"
        )
        await changed()
    }
}
