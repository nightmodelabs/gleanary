import CoreFoundation
import Foundation
import ImageIO

actor IngestionEngine {
    let store: DocumentStore
    let artifacts: ArtifactStore
    let fetcher: any SourceFetching
    let youtube: any YouTubeProviding
    let transcriber: any AudioTranscribing
    /// Follows the AI provider setting; see `useSummarizer`.
    private(set) var summarizer: any LinkSummarizing
    /// False until the app has chosen a summarizer from the AI provider setting, so a capture
    /// made while ChatGPT is still loading doesn't quietly use Apple Intelligence.
    private var summarizerChosen: Bool
    let screenshotter: (any PageScreenshotting)?
    let pageRenderer: (any PageRendering)?
    /// Read at each capture, so Settings changes apply to the next one.
    let settings: @Sendable () -> CaptureSettings
    /// How close found links are to the library, for the eager acquisition levels.
    let relevance: any LinkRelevanceScoring
    /// Reads the wording around found links; cue phrases decide without one.
    private(set) var recommendationJudge: (any RecommendationJudging)?
    /// robots.txt for automatic saves.
    let robots: RobotsPolicy
    /// The least time between automatic fetches from one site, unless it asks for more.
    var automaticSpacing: Duration = .seconds(2)
    private var lastAutomaticFetch: [String: ContinuousClock.Instant] = [:]
    private let pdf = PDFExtractor()
    private var running = false
    private var active: (UUID, Task<UUID, any Error>)?

    init(
        store: DocumentStore,
        fetcher: any SourceFetching = SourceFetcher(),
        youtube: any YouTubeProviding = NativeYouTubeProvider(),
        transcriber: any AudioTranscribing = LocalTranscriber(),
        summarizer: any LinkSummarizing = AppleLinkSummarizer(),
        summarizerChosen: Bool = true,
        screenshotter: (any PageScreenshotting)? = nil,
        pageRenderer: (any PageRendering)? = nil,
        settings: @escaping @Sendable () -> CaptureSettings = { CaptureSettings() },
        relevance: (any LinkRelevanceScoring)? = nil,
        recommendationJudge: (any RecommendationJudging)? = nil
    ) {
        self.recommendationJudge = recommendationJudge
        robots = RobotsPolicy(fetcher: fetcher)
        self.store = store
        self.settings = settings
        self.relevance = relevance ?? LibraryRelevance(search: SearchEngine(store: store))
        self.summarizerChosen = summarizerChosen
        self.fetcher = fetcher
        self.youtube = youtube
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.screenshotter = screenshotter
        self.pageRenderer = pageRenderer
        artifacts = ArtifactStore(root: store.url.deletingLastPathComponent()
            .appending(path: "artifacts"))
    }

    /// Whether summaries stay on the Mac.
    var summarizerIsLocal: Bool {
        summarizer is AppleLinkSummarizer
    }

    func useRecommendationJudge(_ judge: (any RecommendationJudging)?) {
        recommendationJudge = judge
    }

    private func spaceAutomaticFetch(_ job: CaptureJob) async {
        if (job.request.acquisitionHop ?? 0) > 0 {
            await spaceAutomaticFetch(job.request.url)
        }
    }

    /// Waits so automatic fetches from one site are spaced out, as long as its robots.txt
    /// Crawl-delay asks, up to 30 seconds.
    func spaceAutomaticFetch(_ url: URL?) async {
        guard let url, let host = url.host(percentEncoded: false).map(AcquisitionScorer.domain)
        else { return }
        let delay = await robots.rules(for: url).crawlDelay.map { Duration.seconds(min(30, $0)) }
        let spacing = max(automaticSpacing, delay ?? .zero)
        if let last = lastAutomaticFetch[host] {
            let wait = spacing - (ContinuousClock.now - last)
            if wait > .zero {
                try? await Task.sleep(for: wait)
            }
        }
        lastAutomaticFetch[host] = .now
    }

    /// Later summaries use this summarizer, such as ChatGPT once it's enabled.
    func useSummarizer(_ summarizer: any LinkSummarizing) {
        self.summarizer = summarizer
        summarizerChosen = true
    }

    /// The provider's summarizer once chosen; after 20 seconds, whichever is set.
    private func chosenSummarizer() async throws -> any LinkSummarizing {
        for _ in 0 ..< 200 where !summarizerChosen {
            try await Task.sleep(for: .milliseconds(100))
        }
        return summarizer
    }

    func enqueue(
        _ input: String,
        deduplicate: Bool = true,
        summarizeLink: Bool = true,
        replacing noteID: UUID? = nil,
        referencing referenceID: UUID? = nil
    ) async throws -> UUID {
        if let url = try CaptureRequest.classify(input) {
            var request = try CaptureRequest.webLink(url, referencing: referenceID)
            request.summarizeLink = summarizeLink
            request.replacingNoteID = noteID
            return try await store.enqueueCapture(
                request,
                deduplicate: deduplicate && noteID == nil
            )
        }
        guard input.utf8.count <= CaptureLimits.markdownBytes else { throw CaptureError.tooLarge }
        let artifact = try await artifacts.retain(
            Data(input.utf8),
            kind: .markdown,
            name: "Pasted text.md"
        )
        return try await store.enqueueCapture(CaptureRequest(
            artifact: artifact,
            identity: "text:" + artifact.hash,
            label: Self.title(input)
        ), deduplicate: deduplicate)
    }

    /// Fetches the page behind an imported bookmark and upgrades its note in place. `summarize`
    /// false skips the summary even when Capture settings write one.
    func enqueueBookmarkFetch(noteID: UUID, summarize: Bool = true) async throws -> UUID? {
        guard let url = try await store.unfetchedBookmarkURL(noteID: noteID) else { return nil }
        return try await enqueue(url.absoluteString, summarizeLink: summarize, replacing: noteID)
    }

    func enqueueFile(_ url: URL) async throws -> UUID {
        let artifact = try await artifacts.retainFile(url)
        return try await store.enqueueCapture(CaptureRequest(
            artifact: artifact,
            identity: "file:" + artifact.hash,
            label: artifact.name
        ))
    }

    /// `deferAcquisition` leaves links chosen for automatic saving approved but unqueued, so
    /// a caller waiting on this capture can save them after replying; see
    /// `queueApprovedAcquisitions`.
    func run(
        changed: @escaping @Sendable () async -> Void = {},
        deferAcquisition: Bool = false
    ) async throws {
        guard !running else { return }
        running = true
        defer { running = false; active = nil }
        while !Task.isCancelled, let job = try await store.claimCapture() {
            await spaceAutomaticFetch(job)
            let task = Task { try await self.process(job, changed: changed) }
            active = (job.id, task)
            await changed()
            do {
                let noteID = try await withTaskCancellationHandler {
                    try await task.value
                } onCancel: {
                    task.cancel()
                }
                await followLinks(from: job, noteID: noteID, deferred: deferAcquisition)
            } catch is CancellationError {
                try await store.cancelCapture(job.id, token: job.token)
            } catch let error as MediaError {
                try await store.failCapture(
                    job,
                    error: error.localizedDescription,
                    recovery: error.recovery,
                    retryable: CaptureFailure.retryable(error)
                )
            } catch let error as LinkSummaryError {
                try await store.failCapture(job, error: error.localizedDescription)
            } catch let error as SummaryFailure {
                try await store.failCapture(job, error: error.localizedDescription)
            } catch let error as ChatGPTError {
                try await store.failCapture(job, error: error.localizedDescription)
            } catch let error as CaptureError {
                // Only a capture you asked for opens the browser; an automatic one just fails.
                try await store.failCapture(
                    job, error: error.localizedDescription,
                    recovery: job.request.automatic == true ? nil : error.recovery,
                    retryable: error.retryable
                )
            } catch {
                try await store.failCapture(
                    job,
                    error: CaptureError.extractionFailed.localizedDescription,
                    retryable: true
                )
            }
            active = nil
            await changed()
        }
        await upkeep(changed: changed)
    }

    /// After the queue drains: probation, cheap when nothing is due, and suggestion titles.
    private func upkeep(changed: @Sendable () async -> Void) async {
        let expired = await (try? store.expireUnusedAcquisitions()) ?? 0
        let titled = await titleSuggestions()
        if expired + titled > 0 {
            await changed()
        }
    }

    func cancel(_ id: UUID) async throws {
        try await store.cancelCapture(id)
        if active?.0 == id {
            active?.1.cancel()
        }
    }

    func createSummaryNote(
        _ document: MarkdownDocument,
        createAnother: Bool = false
    ) async throws -> MarkdownDocument {
        let source = try await store.summaryNoteInput(for: document, createAnother: createAnother)
        let summarizer = try await chosenSummarizer()
        let summary = try await summarizer.summarize(source.document)
        try Task.checkCancellation()
        return try await store.publishSummaryNote(
            summary, for: document, captureID: source.id, engine: summarizer.engineName,
            createAnother: createAnother
        )
    }

    func summarizeExisting(_ document: MarkdownDocument) async throws -> MarkdownDocument {
        let source = try await store.summarySource(for: document)
        let summarizer = try await chosenSummarizer()
        let summary = try await summarizer.summarize(source.document)
        try Task.checkCancellation()
        let artifact = try await artifacts.retain(
            Data(summary.utf8), kind: .markdown, name: "Summary.md",
            sourceURL: source.document.sourceURL
        )
        do {
            return try await store.publishSummary(
                summary,
                artifact: artifact,
                for: document,
                captureID: source.id,
                engine: summarizer.engineName
            )
        } catch {
            try? await store.queueArtifactCleanup(artifact)
            throw error
        }
    }

    private func process(
        _ job: CaptureJob,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> UUID {
        var acquired = try await acquireOriginals(job, changed: changed)
        guard let primary = acquired.first else { throw CaptureError.missingArtifact }
        var document: CapturedDocument
        var faviconURL: URL?
        var shareImageURL: URL?
        try await store.checkpoint(job, acquired: acquired, progress: "Extracting text")
        await changed()
        switch primary.kind {
        case .capture:
            document = try await JSONDecoder().decode(
                CapturedDocument.self,
                from: artifacts.data(for: primary)
            )
        case .audio:
            guard let url = job.request.url,
                  let target = try MediaTarget.parse(url) else { throw MediaError.invalidLink }
            acquired = try await transcribeAudio(
                target,
                job: job,
                acquired: acquired,
                changed: changed
            )
            guard let snapshot = acquired.first else { throw CaptureError.missingArtifact }
            document = try await JSONDecoder().decode(
                CapturedDocument.self,
                from: artifacts.data(for: snapshot)
            )
        case .json, .image:
            throw CaptureError.unsupported
        case .pdf:
            document = try await extractPDF(primary, job: job, acquired: acquired, changed: changed)
        case .html:
            document = try await extractHTML(
                primary,
                job: job,
                acquired: &acquired,
                faviconURL: &faviconURL,
                shareImageURL: &shareImageURL,
                changed: changed
            )
            try await screenshot(primary, job: job, acquired: &acquired, changed: changed)
            try await shareImage(shareImageURL, job: job, acquired: &acquired)
        case .text, .markdown:
            let text = try await Self.decodeText(
                artifacts.data(for: primary),
                encoding: primary.textEncoding
            )
            let heading = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let title = if job.request.identity.hasPrefix("text:") {
                Self.title(text)
            } else if primary.kind == .markdown, heading.hasPrefix("# ") {
                Self.title(heading)
            } else {
                primary.name
            }
            document = CapturedDocument(
                title: title,
                sourceURL: primary.sourceURL,
                coverage: .note,
                blocks: [ContentBlock(
                    kind: .paragraph,
                    markdown: text,
                    text: text,
                    method: primary.kind == .markdown ? "markdown" : "text"
                )]
            )
        }
        try await videoThumbnail(job, acquired: &acquired)
        try Task.checkCancellation()
        try document.validate()
        // The summarizer can switch while this capture awaits; credit the one that wrote it.
        let summaryEngine = summarizer.engineName
        let (summary, summaryNote) = try await summaryOrNote(
            job, document: document, acquired: &acquired, changed: changed
        )
        try await store.checkpoint(job, acquired: acquired, progress: "Finishing capture")
        await changed()
        let favicon = await FaviconLoader(fetcher: fetcher).load(
            source: primary.sourceURL ?? document.sourceURL ?? job.request.url,
            preferred: faviconURL
        )
        var preview: Data?
        if let still = Self.previewStill(acquired) {
            preview = try? await PagePreview.thumbnail(
                artifacts.data(for: still), centered: still.name != Self.screenshotName
            )
        }
        try Task.checkCancellation()
        return try await store.publishCapture(
            job,
            document: document,
            artifacts: acquired,
            favicon: favicon,
            summary: summary,
            summaryNote: summaryNote,
            summaryEngine: summaryEngine,
            previewImage: preview
        )
    }

    /// A failed summary never costs the capture: the full text is saved, the reason is noted on
    /// the capture, and the item offers Generate summary later.
    private func summaryOrNote(
        _ job: CaptureJob,
        document: CapturedDocument,
        acquired: inout [Artifact],
        changed: @escaping @Sendable () async -> Void
    ) async throws -> (summary: String?, note: String?) {
        do {
            return try await (
                summarize(job, document: document, acquired: &acquired, changed: changed), nil
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return (nil, error.localizedDescription)
        }
    }

    private func summarize(
        _ job: CaptureJob,
        document: CapturedDocument,
        acquired: inout [Artifact],
        changed: @escaping @Sendable () async -> Void
    ) async throws -> String? {
        guard job.request.url != nil, job.request.summarizeLink != false,
              settings().summarize else { return nil }
        var summary: String?
        let summarizer = try await chosenSummarizer()
        let engine = summarizer.engineName
        try await store.checkpoint(
            job,
            acquired: acquired,
            progress: engine.isEmpty ? Self.summarizingProgress
                : "\(Self.summarizingProgress) with \(engine)"
        )
        await changed()
        do {
            summary = try await summarizer.summarize(document)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as LinkSummaryError {
            throw SummaryFailure(engine: engine, underlying: error)
        } catch let error as ChatGPTError {
            // Sign-in and usage problems explain themselves better than a generic failure.
            throw error
        } catch {
            throw LinkSummaryError.failed
        }
        guard let summary, !summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              summary.utf8.count <= CaptureLimits.markdownBytes
        else {
            throw LinkSummaryError.failed
        }
        let artifact = try await artifacts.retain(
            Data(summary.utf8), kind: .markdown,
            name: "Summary.md", sourceURL: document.sourceURL ?? job.request.url
        )
        acquired.append(artifact)
        try await store.checkpoint(
            job,
            acquired: acquired,
            progress: "Summary retained locally"
        )
        return summary
    }

    static let videoThumbnailName = "Thumbnail.jpg"
    /// A capture's progress while it writes the summary, so the item's Summarize button can
    /// show the same state.
    static let summarizingProgress = "Writing a brief summary"
    static let screenshotName = "Screenshot.jpg"
    static let heroName = "Hero.jpg"
    static let shareImageName = "Share image"

    /// The library preview: the page's own share card, else the hero cut from its screenshot,
    /// else a video still, else the top of the screenshot.
    static func previewStill(_ acquired: [Artifact]) -> Artifact? {
        let stills = acquired.filter { $0.kind == .image }
        return [shareImageName, heroName, videoThumbnailName].lazy
            .compactMap { name in stills.first { $0.name == name } }.first ?? stills.first
    }

    /// Best effort: the page's share card, kept only when it's a wide, detailed image rather
    /// than a square logo or a tiny icon.
    private func shareImage(
        _ url: URL?,
        job: CaptureJob,
        acquired: inout [Artifact]
    ) async throws {
        guard let url, settings().previews,
              !acquired.contains(where: { $0.name == Self.shareImageName })
        else { return }
        let data: Data
        do { data = try await fetcher.fetch(url).data }
        catch is CancellationError { throw CancellationError() }
        catch { return }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
              as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width >= 600, height > 0,
              (1.3 ... 2.6).contains(Double(width) / Double(height)) else { return }
        try Task.checkCancellation()
        let artifact = try await artifacts.retain(
            data, kind: .image, name: Self.shareImageName, sourceURL: url
        )
        acquired.append(artifact)
        try await store.checkpoint(
            job,
            acquired: acquired,
            progress: "Share image retained locally"
        )
    }

    /// Best effort: a YouTube video keeps its thumbnail as media and as its library preview.
    /// The largest size doesn't exist for every video, so the always-present one follows it.
    private func videoThumbnail(_ job: CaptureJob, acquired: inout [Artifact]) async throws {
        guard settings().previews, let url = job.request.url,
              let target = try? MediaTarget.parse(url),
              target.site == .youtube,
              !acquired.contains(where: { $0.kind == .image }) else { return }
        for size in ["maxresdefault", "hqdefault"] {
            guard let still = URL(string: "https://i.ytimg.com/vi/\(target.id)/\(size).jpg")
            else { continue }
            let data: Data
            do { data = try await fetcher.fetch(still).data }
            catch is CancellationError { throw CancellationError() }
            catch { continue }
            // YouTube answers a missing size with a tiny grey placeholder.
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any],
                  (properties[kCGImagePropertyPixelWidth] as? Int ?? 0) >= 320 else { continue }
            try Task.checkCancellation()
            let artifact = try await artifacts.retain(
                data, kind: .image, name: Self.videoThumbnailName, sourceURL: still
            )
            acquired.append(artifact)
            try await store.checkpoint(
                job,
                acquired: acquired,
                progress: "Thumbnail retained locally"
            )
            return
        }
    }

    /// Best effort: a page that will not render still yields its extracted text.
    private func screenshot(
        _ primary: Artifact,
        job: CaptureJob,
        acquired: inout [Artifact],
        changed: @escaping @Sendable () async -> Void
    ) async throws {
        guard let screenshotter, settings().previews, job.request.url != nil,
              let url = primary.sourceURL,
              !acquired.contains(where: { $0.kind == .image }) else { return }
        try await store.checkpoint(job, acquired: acquired, progress: "Capturing a page screenshot")
        await changed()
        let shot: PageScreenshot
        do {
            shot = try await screenshotter.capture(url)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            return
        }
        try Task.checkCancellation()
        let artifact = try await artifacts.retain(
            shot.image, kind: .image, name: Self.screenshotName, sourceURL: url
        )
        acquired.append(artifact)
        // The hero as the page lays it out, else where attention goes in its first screens.
        if let top = shot.heroTop ?? PagePreview.salientTop(shot.image),
           let hero = PagePreview.thumbnail(shot.image, top: top)
        {
            try await acquired.append(artifacts.retain(
                hero, kind: .image, name: Self.heroName, sourceURL: url
            ))
        }
        try await store.checkpoint(job, acquired: acquired, progress: "Screenshot retained locally")
    }

    private func acquireOriginals(
        _ job: CaptureJob,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> [Artifact] {
        guard job.acquired.isEmpty else { return job.acquired }
        guard let url = job.request.url else { throw CaptureError.missingArtifact }
        let acquired: [Artifact] = if let target = try MediaTarget.parse(url) {
            try await acquireMedia(target, job: job, changed: changed)
        } else {
            try await [downloadOrRender(url, job: job, changed: changed)]
        }
        try await store.checkpoint(job, acquired: acquired, progress: "Original retained locally")
        await changed()
        return acquired
    }

    private func extractHTML(
        _ primary: Artifact, job: CaptureJob, acquired: inout [Artifact],
        faviconURL: inout URL?,
        shareImageURL: inout URL?,
        changed: @escaping @Sendable () async -> Void
    ) async throws -> CapturedDocument {
        let data = try await artifacts.data(for: primary)
        let html = try Self.decodeText(data, encoding: primary.textEncoding, html: true)
        let analysis = try HTMLExtractor.analyze(html, url: primary.sourceURL)
        faviconURL = analysis.faviconURL
        shareImageURL = analysis.shareImageURL
        guard let pdfURL = analysis.pdfURL else {
            return try await HTMLExtractor.extract(analysis, url: primary.sourceURL)
        }
        do {
            let pdfArtifact: Artifact
            if let retained = acquired.dropFirst().first(where: { $0.kind == .pdf }) {
                pdfArtifact = retained
            } else {
                pdfArtifact = try await download(pdfURL)
                guard pdfArtifact.kind == .pdf else { throw CaptureError.invalidPDF }
                acquired.append(pdfArtifact)
                try await store.checkpoint(
                    job,
                    acquired: acquired,
                    progress: "Full-text PDF retained locally"
                )
            }
            var document = try await extractPDF(
                pdfArtifact,
                job: job,
                acquired: acquired,
                changed: changed
            )
            if !analysis.title.isEmpty {
                document.title = analysis.title
            }
            if !analysis.authors.isEmpty {
                document.authors = analysis.authors
            }
            document.published = analysis.published
            document.identifier = analysis.identifier
            return document
        } catch is CancellationError { throw CancellationError() }
        catch {
            try Task.checkCancellation()
            var document = try await HTMLExtractor.extract(analysis, url: primary.sourceURL)
            document.coverage = analysis.academic ? .abstractOnly : .partial
            document.warnings
                .append(
                    "The linked full-text PDF could not be acquired or extracted. Only the landing-page text was captured."
                )
            return document
        }
    }

    /// A site that refuses a plain request, such as one behind Cloudflare's check, may let a
    /// browser through. For a capture you asked for, the page loads in a private, unseen
    /// browser; if its check wants a person, the capture offers the browser window instead.
    private func downloadOrRender(
        _ url: URL, job: CaptureJob, changed: @escaping @Sendable () async -> Void
    ) async throws -> Artifact {
        do {
            return try await download(url)
        } catch let CaptureError.http(status) where Self.browserMayPass(status) {
            guard job.request.automatic != true, let pageRenderer else {
                throw CaptureError.http(status)
            }
            try await store.checkpoint(job, acquired: [], progress: "Opening in a private browser")
            await changed()
            guard let page = try await pageRenderer.render(url) else {
                throw CaptureError.needsVerification
            }
            return try await retainRendered(page)
        }
    }

    nonisolated static func browserMayPass(_ status: Int) -> Bool {
        [401, 403, 429, 503].contains(status)
    }

    /// A page you loaded in the private browser, kept as its HTML original.
    func retainRendered(_ page: RenderedPage) async throws -> Artifact {
        try Self.requireDocumentSource(page.url)
        return try await artifacts.retain(
            Data(page.html.utf8), kind: .html,
            name: Self.artifactName(page.url),
            sourceURL: page.url, textEncoding: "utf-8"
        )
    }

    /// Resumes a capture with the page you loaded in the private browser window.
    func submitRenderedPage(_ page: RenderedPage, job: CaptureJob) async throws {
        let artifact = try await retainRendered(page)
        try Task.checkCancellation()
        try await store.resumeMedia(job, artifacts: [artifact])
    }

    private func download(_ url: URL) async throws -> Artifact {
        try Self.requireDocumentSource(url)
        let fetched = try await fetcher.fetch(url)
        try Self.requireDocumentSource(fetched.url)
        let kind = try ArtifactStore.kind(
            data: fetched.data,
            extension: fetched.url.pathExtension,
            mime: fetched.mime
        )
        return try await artifacts.retain(
            fetched.data,
            kind: kind,
            name: Self.artifactName(fetched.url),
            sourceURL: fetched.url,
            textEncoding: fetched.encoding
        )
    }

    private func extractPDF(
        _ artifact: Artifact,
        job: CaptureJob,
        acquired: [Artifact],
        changed: @escaping @Sendable () async -> Void
    ) async throws -> CapturedDocument {
        let data = try await artifacts.data(for: artifact)
        var document = try await pdf
            .extract(data, name: artifact.name, url: artifact.sourceURL) { [store] message in
                try await store.checkpoint(job, acquired: acquired, progress: message)
                await changed()
            }
        try document.validate()
        return document
    }

    /// A site's root path is "/", which names nothing, so it falls back to the host.
    nonisolated static func artifactName(_ url: URL) -> String {
        let name = url.lastPathComponent
        return name.isEmpty || name == "/" ? url.host ?? "Web capture" : name
    }

    private static func requireDocumentSource(_ url: URL) throws {
        guard try MediaTarget.parse(url) == nil else { throw CaptureError.unsupported }
    }

    private static func title(_ text: String) -> String {
        let first = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? "Untitled capture"
        let cleaned = first.trimmingCharacters(in: CharacterSet(charactersIn: "# ")).prefix(80)
        return cleaned.isEmpty ? "Untitled capture" : String(cleaned)
    }

    static func decodeText(
        _ data: Data,
        encoding name: String? = nil,
        html: Bool = false
    ) throws -> String {
        var encoding = name
        if encoding == nil, html {
            let prefix = String(decoding: data.prefix(4096), as: UTF8.self)
            if let match = prefix.range(
                of: #"charset\s*=\s*["']?([A-Za-z0-9_-]+)"#,
                options: [.regularExpression, .caseInsensitive]
            ) {
                encoding = prefix[match].split(separator: "=").last?
                    .trimmingCharacters(in: CharacterSet(charactersIn: " \"'"))
            }
        }
        if let encoding {
            let cfEncoding = CFStringConvertIANACharSetNameToEncoding(encoding as CFString)
            if cfEncoding != kCFStringEncodingInvalidId,
               let text = String(
                   data: data,
                   encoding: String
                       .Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cfEncoding))
               )
            {
                return text
            }
        }
        if data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF]), let text = String(
            data: data,
            encoding: .utf16
        ) {
            return text
        }
        guard let text = String(data: data, encoding: .utf8) else { throw CaptureError.unsupported }
        return text.hasPrefix("\u{FEFF}") ? String(text.dropFirst()) : text
    }
}
