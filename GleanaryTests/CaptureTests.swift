import AppKit
import CoreText
import Foundation
@testable import Gleanary
import GRDB
import Testing

private struct CaptureFixture {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let store: DocumentStore
    init() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = DocumentStore(url: root.appending(path: "library.sqlite"))
    }

    func clean() {
        try? FileManager.default.removeItem(at: root)
    }
}

private actor FixtureFetcher: SourceFetching {
    var responses: [String: DownloadedContent] = [:]
    var calls = 0
    var requestedURLs: [URL] = []
    func set(_ url: URL, data: Data, mime: String, finalURL: URL? = nil) {
        responses[url.absoluteString] = DownloadedContent(
            data: data,
            url: finalURL ?? url,
            mime: mime,
            encoding: "utf-8"
        )
    }

    func fetch(_ url: URL) throws -> DownloadedContent {
        calls += 1
        requestedURLs.append(url)
        guard let response = responses[url.absoluteString] else { throw CaptureError.http(403) }
        return response
    }
}

private actor HeldFetcher: SourceFetching {
    private var held: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<Void, Never>?
    private var first = true
    func fetch(_ url: URL) async throws -> DownloadedContent {
        if first {
            first = false
            await withCheckedContinuation { held = $0; waiter?.resume(); waiter = nil }
            try Task.checkCancellation()
        }
        return DownloadedContent(
            data: Data("A retried capture completes without an old worker cancelling it.".utf8),
            url: url,
            mime: "text/plain",
            encoding: nil
        )
    }

    func waitUntilHeld() async {
        if held == nil {
            await withCheckedContinuation { waiter = $0 }
        }
    }

    func release() {
        held?.resume(); held = nil
    }
}

/// A browser that gets past the site's check, or one whose check waits for a person.
private actor FixtureRenderer: PageRendering {
    let html: String?
    var calls = 0
    init(html: String?) {
        self.html = html
    }

    func render(_ url: URL) -> RenderedPage? {
        calls += 1
        return html.map { RenderedPage(html: $0, url: url) }
    }
}

private struct FixtureResolver: HostResolving {
    let addresses: [String]
    func addresses(for _: String) async throws -> [String] {
        addresses
    }
}

@Suite(.serialized)
struct CaptureTests {
    private static let prose = "Local notebooks preserve knowledge without requiring a remote account. A durable database keeps revisions and original sources together, while a separate search index can be rebuilt on this computer. The reader can check each retrieved passage against the retained original, rather than trusting an automated summary."

    private static func article(_ extra: String = "") -> String {
        """
        <!doctype html><html><head><title>Durable knowledge</title><meta name="author" content="Synthetic Writer"></head>
        <body><nav>Navigation noise</nav><article><h1>Durable knowledge</h1>
        <p>\(prose)</p><h2>Source fidelity</h2><p>\(prose)</p>\(extra)
        <p>Keep a <a href="/reference">reference link</a> alongside the original capture.</p></article></body></html>
        """
    }

    @Test func classifiesInputAndPreservesURLSemantics() throws {
        #expect(try CaptureRequest.classify("a thought about my notes") == nil)
        #expect(try CaptureRequest.classify("# A heading\n\nSome text") == nil)
        #expect(try CaptureRequest.classify("example.com/article")?.scheme == "https")
        let url = try #require(URL(string: "HTTPS://Example.com:443/paper?q=a%2Bb#section"))
        #expect(try CaptureRequest.canonicalURL(url) == "https://example.com/paper?q=a%2Bb")
        #expect(throws: CaptureError.self) { try CaptureRequest.classify("file:///etc/passwd") }
        #expect(throws: CaptureError.self) {
            try CaptureRequest
                .canonicalURL(#require(URL(string: "https://name:password@example.com/")))
        }
    }

    @Test func classifiesNegotiatedMarkdownAtAnExtensionlessURL() throws {
        let data = Data("# Omnigraph\n\nA page served as Markdown.".utf8)
        #expect(try ArtifactStore.kind(data: data, mime: "text/markdown") == .markdown)
        #expect(try ArtifactStore.kind(data: data, mime: "text/x-markdown") == .markdown)
        #expect(throws: CaptureError.self) { try ArtifactStore.kind(data: data, mime: "application/json") }
    }

    @Test func rejectsPrivateLiteralAndResolvedAddresses() async throws {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.4.1",
            "172.20.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "2001:db8::1",
        ] {
            #expect(!URLPolicy.isPublicAddress(address))
        }
        #expect(URLPolicy.isPublicAddress("93.184.216.34"))
        #expect(URLPolicy.isPublicAddress("2606:4700:4700::1111"))
        let mixed = URLPolicy(resolver: FixtureResolver(addresses: ["93.184.216.34", "127.0.0.1"]))
        await #expect(throws: CaptureError.self) {
            try await mixed.validate(#require(URL(string: "https://public-looking.example/")))
        }
        let publicPolicy = URLPolicy(resolver: FixtureResolver(addresses: ["93.184.216.34"]))
        await #expect(throws: CaptureError.self) {
            try await publicPolicy.validate(#require(URL(string: "https://example.com:8080/")))
        }
    }

    @Test func textCaptureIsDurableAndSearchable() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: FixtureFetcher(),
            summarizer: FixtureLinkSummarizer()
        )
        let text = "# Local memory\n\nZebracobalt notes are retained on this computer."
        _ = try await engine.enqueue(text)
        try await engine.run()
        let documents = try await fixture.store.documents()
        #expect(documents.count == 1)
        #expect(documents.first?.markdown == text)
        let snapshot = try await fixture.store.searchSnapshot(
            query: "zebracobalt",
            candidateLimit: 10
        )
        #expect(snapshot.keywordIDs.count == 1)
        let citation = try #require(snapshot.passages.first?.citation)
        let detail = try await fixture.store.captureDetail(id: citation.captureID)
        #expect(detail.document.markdown == text)
        #expect(try await engine.artifacts
            .data(for: #require(detail.artifacts.first)) == Data(text.utf8))
        let duplicateID = try await engine.enqueue(text)
        let duplicate = try #require(try await fixture.store.captureJobs()
            .first { $0.id == duplicateID })
        #expect(duplicate.state == .ready)
        #expect(duplicate.noteID == documents.first?.id)
        #expect(duplicate.progress == "Already captured; existing note kept")
        #expect(duplicate.savedTitle == documents.first?.name)
        #expect(try await fixture.store.claimCapture() == nil)
        try await engine.run()
        #expect(try await fixture.store.documents().count == 1)
        let reopened = DocumentStore(url: fixture.store.url)
        #expect(try await reopened.open(seed: []).count == 1)
    }

    @Test func permanentDeletePurgesHistoryAndUnsharedFilesWithoutResurrection() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let artifacts = ArtifactStore(root: fixture.root.appending(path: "artifacts"))
        let shared = try await artifacts.retain(
            Data("Shared original fixture".utf8),
            kind: .text,
            name: "shared.txt"
        )
        let unique = try await artifacts.retain(
            Data("Unique original fixture".utf8),
            kind: .text,
            name: "unique.txt"
        )
        let content = CapturedDocument(
            title: "Synthetic capture",
            coverage: .note,
            blocks: [ContentBlock(
                kind: .paragraph,
                markdown: "Synthetic retained knowledge.",
                text: "Synthetic retained knowledge.",
                method: "fixture"
            )]
        )
        _ = try await fixture.store.enqueueCapture(CaptureRequest(
            identity: "fixture:first",
            label: "First"
        ))
        let firstJob = try #require(await fixture.store.claimCapture())
        let firstID = try await fixture.store.publishCapture(
            firstJob,
            document: content,
            artifacts: [shared, unique]
        )
        let firstCapture = try #require(await fixture.store.captureID(for: firstID))
        _ = try await fixture.store.enqueueCapture(CaptureRequest(
            identity: "fixture:second",
            label: "Second"
        ))
        let secondJob = try #require(await fixture.store.claimCapture())
        _ = try await fixture.store.publishCapture(
            secondJob,
            document: content,
            artifacts: [shared]
        )
        let first = try #require(await fixture.store.documents().first { $0.id == firstID })
        var draft = first
        draft.markdown += " Changed text."
        _ = try await fixture.store.save(draft)
        await #expect(throws: GleanaryError.self) {
            try await fixture.store.permanentlyRemove(first)
        }
        #expect(try await fixture.store.documents().count == 2)
        let current = try #require(await fixture.store.documents().first { $0.id == firstID })
        _ = try await fixture.store.enqueueCapture(CaptureRequest(
            identity: "fixture:first",
            label: "Pending recapture"
        ), deduplicate: false)
        let stale = try #require(await fixture.store.claimCapture())
        let removed = try await fixture.store.permanentlyRemove(current)
        #expect(removed.contains(stale.id))
        #expect(try await fixture.store.history(for: firstID).isEmpty)
        await #expect(throws: GleanaryError.self) {
            try await fixture.store.captureDetail(id: firstCapture)
        }
        await #expect(throws: GleanaryError.self) { try await fixture.store.save(current) }
        await #expect(throws: CaptureError.self) { try await fixture.store.publishCapture(
            stale,
            document: content,
            artifacts: [unique]
        ) }
        // Cleanup survives a restart between the database transaction and physical file removal.
        let reopened = DocumentStore(url: fixture.store.url)
        _ = try await reopened.open(seed: [])
        #expect(try await reopened.hasPendingArtifactDeletions())
        #expect(try await reopened.collectDeletedArtifacts() == 0)
        #expect(try await artifacts.data(for: shared) == Data("Shared original fixture".utf8))
        await #expect(throws: CaptureError.self) { try await artifacts.data(for: unique) }
        let remaining = try #require(await reopened.documents().first)
        _ = try await reopened.permanentlyRemove(remaining)
        try await reopened.collectDeletedArtifacts()
        await #expect(throws: CaptureError.self) { try await artifacts.data(for: shared) }
        #expect(try await reopened.documents().isEmpty)
        #expect(try await reopened.captureJobs().isEmpty)
        let db = try DatabaseQueue(path: fixture.store.url.path)
        try await db.read { db in
            for table in [
                "notes",
                "note_revisions",
                "local_changes",
                "captures",
                "content_blocks",
                "sources",
                "artifacts",
                "passages",
                "index_jobs",
                "artifact_deletions",
            ] {
                let count = try Int.fetchOne(db, sql: "SELECT count(*) FROM \(table)")
                #expect(count == 0)
            }
            let markers = try Int.fetchOne(db, sql: "SELECT count(*) FROM purged_notes")
            #expect(markers == 2)
        }
    }

    @Test func artifactRetentionSurvivesInputRemovalAndRepairsDamage() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let engine = IngestionEngine(store: fixture.store, summarizer: FixtureLinkSummarizer())
        let file = fixture.root.appending(path: "local.md")
        let data = Data("A synthetic document kept for reliable local retrieval.".utf8)
        try data.write(to: file)
        let id = try await engine.enqueueFile(file)
        try FileManager.default.removeItem(at: file)
        let job = try #require(await fixture.store.captureJobs().first)
        let artifact = try #require(job.acquired.first)
        let storedURL = try await engine.artifacts.fileURL(artifact.hash)
        try Data("damaged".utf8).write(to: storedURL)
        try await engine.run()
        #expect(try await fixture.store.documents().isEmpty)
        #expect(try await fixture.store.captureJobs().first?.state == .failed)
        _ = try await engine.artifacts.retain(data, kind: .markdown, name: "local.md")
        try await fixture.store.retryCapture(id)
        try await engine.run()
        #expect(try await fixture.store.documents().count == 1)
        #expect(try await engine.artifacts.data(for: artifact) == data)
        await #expect(throws: CaptureError.self) { try await engine.artifacts.fileURL("../outside")
        }
    }

    @Test func cancellationAndRecoveryRejectStalePublication() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let engine = IngestionEngine(store: fixture.store, summarizer: FixtureLinkSummarizer())
        let id = try await engine.enqueue("A cancelled synthetic capture must never publish.")
        let first = try #require(await fixture.store.claimCapture())
        try await fixture.store.cancelCapture(id)
        let document = CapturedDocument(
            title: "Cancelled",
            coverage: .note,
            blocks: [ContentBlock(kind: .paragraph, markdown: "Text", text: "Text")]
        )
        await #expect(throws: CaptureError.self) { try await fixture.store.publishCapture(
            first,
            document: document,
            artifacts: []
        ) }
        #expect(try await fixture.store.documents().isEmpty)
        try await fixture.store.retryCapture(id)
        let second = try #require(await fixture.store.claimCapture())
        try await fixture.store.recoverCaptureJobs()
        await #expect(throws: CaptureError.self) { try await fixture.store.publishCapture(
            second,
            document: document,
            artifacts: []
        ) }
        try await engine.run()
        #expect(try await fixture.store.documents().count == 1)
    }

    @Test(arguments: [true, false]) @MainActor
    func capturesNonArticleMainTextWithoutSourceScriptsOrPageControls(hasMain: Bool) async throws {
        _ = NSApplication.shared
        let tag = hasMain ? "main" : "div"
        let html = """
        <html><head><title>Testing toolkit</title></head><body>
        <nav>Navigation chrome</nav>
        <\(tag)><h1>Testing toolkit</h1>
        <h2><a href="/models">Choose your testing model.</a></h2>
        <h2><a href="/keys">Bring your own credentials.</a></h2>
        <h2><a href="/pipeline">Run checks in your pipeline.</a></h2>
        <h2><a href="/results">Keep the results on your machine.</a></h2>
        <h2><a href="/guide">Read the guide</a></h2>
        <h2><a href="/example"><code>testFixture("evidencealpha")</code></a></h2>
        <button>Subscribe control</button><div hidden>Hidden fixture</div>
        <script>document.body.innerHTML = 'COMPROMISED'; fetch('https://never-contact.invalid');</script>
        </\(tag)><footer>Footer chrome</footer></body></html>
        """
        let url = try #require(URL(string: "https://fixture.example/landing"))
        let analysis = try HTMLExtractor.analyze(html, url: url)
        var document = try await HTMLExtractor.extract(analysis, url: url)
        try document.validate()
        #expect(document.coverage == .partial)
        #expect(document.blocks.allSatisfy { $0.method == (hasMain ? "html-main" : "html-body") })
        #expect(document.markdown.contains("Choose your testing model"))
        #expect(document.markdown.contains("evidencealpha"))
        #expect(document.markdown.contains("https://fixture.example/guide"))
        #expect(!document.markdown.contains("COMPROMISED"))
        #expect(!document.markdown.contains("Navigation chrome"))
        #expect(!document.markdown.contains("Footer chrome"))
        #expect(!document.markdown.contains("Subscribe control"))
        #expect(!document.markdown.contains("Hidden fixture"))
        #expect(document.warnings.contains { $0.contains("No article was identified") })
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let engine = IngestionEngine(store: fixture.store, fetcher: fetcher)
        let id = try await engine.enqueue(url.absoluteString, summarizeLink: false)
        let claimed = try #require(await fixture.store.claimCapture())
        let original = try await engine.artifacts.retain(
            Data(html.utf8),
            kind: .html,
            name: "landing.html",
            sourceURL: url
        )
        try await fixture.store.checkpoint(
            claimed,
            acquired: [original],
            progress: "Original retained"
        )
        try await fixture.store.failCapture(
            claimed,
            error: CaptureError.extractionFailed.localizedDescription
        )
        try await fixture.store.retryCapture(id)
        try await engine.run()
        #expect(try await fixture.store.captureJob(id: id)?.state == .ready)
        #expect(await fetcher.requestedURLs.allSatisfy { $0 != url })
        let note = try #require(await fixture.store.documents().first)
        #expect(try await SearchEngine(store: fixture.store).search("evidencealpha").matches
            .contains { $0.passage.documentID == note.id })
    }

    @Test @MainActor func emptyClientRenderedPagesDoNotBecomeSuccessfulCaptures() async throws {
        _ = NSApplication.shared
        let analysis = try HTMLExtractor.analyze("""
        <html><head><title>Empty app</title></head><body><main></main>
        <script>document.querySelector('main').textContent = 'Requires page scripts';</script>
        </body></html>
        """, url: nil)
        await #expect {
            _ = try await HTMLExtractor.extract(analysis, url: nil)
        } throws: { error in
            guard case CaptureError.noContent = error else { return false }
            return true
        }
    }

    @Test @MainActor func extractsFaithfulHTMLInRestrictedDOM() async throws {
        _ = NSApplication.shared
        let url = try #require(URL(string: "https://fixture.example/articles/one"))
        let extra = """
        <script>window.compromised = true; fetch('https://never-contact.invalid/');</script>
        <p onclick="alert('bad')">Safe prose is retained without page scripts.</p>
        <pre><code>let value_name = 42;</code></pre><ul><li>First fact</li><li>Second fact</li></ul>
        <table><thead><tr><th>Method</th><th>Value</th></tr></thead><tbody><tr><td>Local</td><td>42</td></tr></tbody></table>
        <iframe src="https://never-contact.invalid/"></iframe><a href="javascript:alert(1)">Unsafe link</a>
        """
        let analysis = try HTMLExtractor.analyze(Self.article(extra), url: url)
        #expect(!analysis.sanitizedHTML.contains("<script"))
        #expect(!analysis.sanitizedHTML.contains("onclick="))
        #expect(!analysis.sanitizedHTML.contains("javascript:"))
        #expect(!analysis.sanitizedHTML.contains("<iframe"))
        var document = try await HTMLExtractor.extract(analysis, url: url)
        try document.validate()
        #expect(document.markdown.contains("https://fixture.example/reference"))
        #expect(document.markdown.contains("value_name"))
        #expect(document.markdown.contains("| Method | Value |"))
        let list = try #require(document.blocks.first { $0.kind == .list })
        #expect(list.text.split(whereSeparator: \.isWhitespace)
            .joined(separator: " ") == "First fact Second fact")
        #expect(document.blocks.contains { $0.kind == .code })
        #expect(document.blocks.contains { $0.section?.contains("Source fidelity") == true })
        #expect(!document.markdown.contains("Navigation noise"))
        #expect(document.authors == ["Synthetic Writer"])
    }

    @Test @MainActor func pageFormulasAreCapturedAsTheirTeX() async throws {
        _ = NSApplication.shared
        let extra = #"""
        <p>KaTeX gives <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>x</mi></mrow><annotation encoding="application/x-tex">a_i^2</annotation></semantics></math></span><span class="katex-html">a i 2</span></span> inline.</p>
        <span class="katex-display"><span class="katex"><span class="katex-mathml"><math display="block"><semantics><mrow></mrow><annotation encoding="application/x-tex">\sum_{k=1}^n k</annotation></semantics></math></span></span></span>
        <p>Wikipedia writes <span class="mwe-math-element"><span class="mwe-math-mathml-inline"><math alttext="{\displaystyle e^{i\pi}}"></math></span><img class="mwe-math-fallback-image-inline" alt="{\displaystyle e^{i\pi}}"></span> this way.</p>
        <p>MathJax 2 keeps <span class="MathJax">junk</span><script type="math/tex">\frac{a}{b}</script> in scripts.</p>
        <p>MathML alone has <math><msup><mi>y</mi><mn>2</mn></msup></math> in it.</p>
        <p>Browser-typeset TeX like $\alpha_1 * \beta_2$ stays unescaped, while 2*3 and_this do not.</p>
        """#
        let analysis = try HTMLExtractor.analyze(Self.article(extra), url: nil)
        let document = try await HTMLExtractor.extract(analysis, url: nil)
        let markdown = document.markdown
        #expect(markdown.contains("KaTeX gives $a_i^2$ inline."))
        #expect(markdown.contains("$$\\sum_{k=1}^n k$$"))
        #expect(markdown.contains("Wikipedia writes $e^{i\\pi}$ this way."))
        #expect(markdown.contains("MathJax 2 keeps $\\frac{a}{b}$ in scripts."))
        #expect(!markdown.contains("junk"))
        #expect(markdown.contains("MathML alone has ${y}^{2}$ in it."))
        #expect(markdown.contains("$\\alpha_1 * \\beta_2$"))
        #expect(markdown.contains("2\\*3 and\\_this"))
        #expect(document.blocks.contains { $0.text.contains("$a_i^2$") })
    }

    @Test @MainActor func refreshDoesNotOverwriteEditsOrResurrectDeletion() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let url = try #require(URL(string: "https://fixture.example/article"))
        await fetcher.set(url, data: Data(Self.article().utf8), mime: "text/html")
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        _ = try await engine.enqueue(url.absoluteString); try await engine.run()
        var edited = try #require(await fixture.store.documents().first)
        let originalID = edited.id
        edited.markdown = "My own annotation with cobaltbeacon."
        edited = try await fixture.store.save(edited)
        let editedSnapshot = try await fixture.store.searchSnapshot(
            query: "cobaltbeacon",
            candidateLimit: 10
        )
        #expect(editedSnapshot.passages.allSatisfy { $0.citation == nil })
        _ = try await engine.enqueue(url.absoluteString); try await engine.run()
        #expect(try await fixture.store.documents() == [edited])
        await fetcher.set(
            url,
            data: Data(Self
                .article("<p>A newly published revision with unique updated findings.</p>").utf8),
            mime: "text/html"
        )
        _ = try await engine.enqueue(url.absoluteString, deduplicate: false); try await engine.run()
        #expect(try await fixture.store.documents().count == 2)
        #expect(try await fixture.store.documents().first(where: { $0.id == originalID })?
            .markdown == edited.markdown)
        try await fixture.store.remove(edited)
        await fetcher.set(url, data: Data(Self.article().utf8), mime: "text/html")
        _ = try await engine.enqueue(url.absoluteString, deduplicate: false); try await engine.run()
        #expect(try await fixture.store.documents().count == 1)
        #expect(try await fixture.store.captureJobs().contains { $0.state == .failed })
    }

    @Test @MainActor func aRefusedPageYouAskedForLoadsInThePrivateBrowser() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture()
        defer { fixture.clean() }
        _ = try await fixture.store.open()
        let renderer = FixtureRenderer(html: Self.article())
        let engine = IngestionEngine(
            store: fixture.store, fetcher: FixtureFetcher(), summarizer: FixtureLinkSummarizer(),
            pageRenderer: renderer
        )
        _ = try await engine.enqueue("https://fixture.example/guarded"); try await engine.run()
        #expect(await renderer.calls == 1)
        #expect(try await fixture.store.documents().first?.name == "Durable knowledge")
    }

    @Test @MainActor func aCheckThatWantsAPersonOffersTheBrowserWindow() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture()
        defer { fixture.clean() }
        _ = try await fixture.store.open()
        let engine = IngestionEngine(
            store: fixture.store, fetcher: FixtureFetcher(), summarizer: FixtureLinkSummarizer(),
            pageRenderer: FixtureRenderer(html: nil)
        )
        _ = try await engine.enqueue("https://fixture.example/guarded"); try await engine.run()
        let failed = try #require(await fixture.store.captureJobs().first)
        #expect(failed.state == .failed && failed.recovery == .browser)
        #expect(failed.error == CaptureError.needsVerification.localizedDescription)

        // The page you loaded in the window resumes the same capture.
        let url = try #require(URL(string: "https://fixture.example/guarded"))
        try await engine.submitRenderedPage(
            RenderedPage(html: Self.article(), url: url),
            job: failed
        )
        try await engine.run()
        #expect(try await fixture.store.documents().first?.name == "Durable knowledge")
    }

    @Test @MainActor func anAutomaticSaveTakesTheSitesRefusal() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture()
        defer { fixture.clean() }
        _ = try await fixture.store.open()
        let renderer = FixtureRenderer(html: Self.article())
        let engine = IngestionEngine(
            store: fixture.store, fetcher: FixtureFetcher(), summarizer: FixtureLinkSummarizer(),
            pageRenderer: renderer
        )
        var request = try CaptureRequest
            .webLink(#require(URL(string: "https://fixture.example/guarded")))
        request.automatic = true
        _ = try await fixture.store.enqueueCapture(request); try await engine.run()
        #expect(await renderer.calls == 0)
        let failed = try #require(await fixture.store.captureJobs().first)
        #expect(failed.state == .failed && failed.recovery == nil)
        #expect(failed.error == CaptureError.http(403).localizedDescription)

        // Trying it again yourself is asking for it.
        try await fixture.store.retryCapture(failed.id)
        try await engine.run()
        #expect(await renderer.calls == 1)
        #expect(try await fixture.store.documents().first?.name == "Durable knowledge")
    }

    @Test @MainActor func resumesRetainedHTMLWithoutDownloadingAgain() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        let url = try #require(URL(string: "https://fixture.example/resume"))
        _ = try await engine.enqueue(url.absoluteString)
        let claimed = try #require(await fixture.store.claimCapture())
        let original = try await engine.artifacts.retain(
            Data(Self.article().utf8),
            kind: .html,
            name: "resume.html",
            sourceURL: url
        )
        try await fixture.store.checkpoint(claimed, acquired: [original], progress: "Downloaded")
        let reopened = DocumentStore(url: fixture.store.url)
        _ = try await reopened.open(seed: [])
        try await reopened.recoverCaptureJobs()
        let resumed = IngestionEngine(
            store: reopened,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        try await resumed.run()
        // Optional icon acquisition must not redownload the retained article.
        #expect(await fetcher.requestedURLs == [URL(string: "https://fixture.example/favicon.ico")])
        #expect(try await reopened.documents().count == 1)
    }

    @Test @MainActor func academicFailureIsExplicitlyPartial() async throws {
        _ = NSApplication.shared
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let url = try #require(URL(string: "https://fixture.example/paper"))
        let html = Self.article().replacingOccurrences(
            of: "</head>",
            with: "<meta name='citation_title' content='Synthetic paper'><meta name='citation_pdf_url' content='/full.pdf'><meta name='citation_doi' content='10.0000/synthetic'></head>"
        )
        await fetcher.set(url, data: Data(html.utf8), mime: "text/html")
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        _ = try await engine.enqueue(url.absoluteString); try await engine.run()
        let note = try #require(await fixture.store.documents().first)
        let id = try #require(await fixture.store.captureID(for: note.id))
        let detail = try await fixture.store.captureDetail(id: id)
        #expect(detail.document.coverage == .abstractOnly)
        #expect(detail.document.identifier == "10.0000/synthetic")
        #expect(detail.document.warnings.contains { $0.contains("could not be acquired") })
        #expect(detail.artifacts.map(\.kind) == [.html, .markdown])
    }

    @Test func academicCaptureRetainsLandingPageAndMainPDF() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let url = try #require(URL(string: "https://fixture.example/paper"))
        let pdfURL = try #require(URL(string: "https://fixture.example/full.pdf"))
        let html = Self.article().replacingOccurrences(
            of: "</head>",
            with: "<meta name='citation_title' content='Synthetic full paper'><meta name='citation_pdf_url' content='/full.pdf'></head>"
        )
        await fetcher.set(url, data: Data(html.utf8), mime: "text/html")
        try await fetcher.set(pdfURL, data: Self.pdfData(), mime: "application/pdf")
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        _ = try await engine.enqueue(url.absoluteString)
        try await engine.run()
        let note = try #require(await fixture.store.documents().first)
        let id = try #require(await fixture.store.captureID(for: note.id))
        let detail = try await fixture.store.captureDetail(id: id)
        #expect(detail.document.coverage == .document)
        #expect(detail.document.title == "Synthetic full paper")
        #expect(detail.document.sourceURL == pdfURL)
        #expect(detail.artifacts.map(\.kind) == [.html, .pdf, .markdown])
        #expect(detail.document.blocks.contains { $0.page == 2 })
    }

    @Test func malformedMediaNeverFallsThroughToGenericArticleCapture() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = FixtureFetcher()
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        await #expect(throws: MediaError.self) {
            try await engine.enqueue("https://www.youtube.com/watch?v=synthetic")
        }
        #expect(await fetcher.calls == 0)
        let shortened = try #require(URL(string: "https://fixture.example/short"))
        await fetcher.set(
            shortened,
            data: Data(Self.article().utf8),
            mime: "text/html",
            finalURL: URL(string: "https://www.youtube.com/watch?v=synthetic")
        )
        _ = try await engine.enqueue(shortened.absoluteString)
        try await engine.run()
        #expect(await fetcher.calls == 1)
        #expect(try await fixture.store.documents().isEmpty)
        #expect(try await fixture.store.captureJobs().allSatisfy { $0.state == .failed })
    }

    @Test func loginAndAmbiguousPDFsAreNotFullText() throws {
        #expect(throws: CaptureError.self) { try HTMLExtractor.analyze(
            "<html><title>Just a moment...</title><body>Verify</body></html>",
            url: nil
        ) }
        #expect(throws: CaptureError.self) { try HTMLExtractor.analyze(
            "<html><title>Account</title><form><input type='password'></form></html>",
            url: nil
        ) }
        let ambiguous = Self
            .article(
                "<a href='https://fixture.example/a.pdf'>PDF</a><a href='https://fixture.example/b.pdf'>Download PDF</a>"
            )
        let result = try HTMLExtractor.analyze(ambiguous, url: nil)
        #expect(result.pdfURL == nil)
        #expect(result.warnings.contains { $0.contains("Multiple") })
    }

    @Test func PDFTextAndPageCitationsSurvivePublication() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let data = try Self.pdfData()
        let file = fixture.root.appending(path: "synthetic.pdf")
        try data.write(to: file)
        let engine = IngestionEngine(store: fixture.store, summarizer: FixtureLinkSummarizer())
        _ = try await engine.enqueueFile(file)
        try await engine.run()
        let snapshot = try await fixture.store.searchSnapshot(
            query: "secondbeacon",
            candidateLimit: 10
        )
        let match = try #require(snapshot.passages.first { $0.text.contains("secondbeacon") })
        #expect(match.citation?.page == 2)
        let detail = try await fixture.store.captureDetail(id: #require(match.citation?.captureID))
        #expect(detail.document.blocks.contains { $0.page == 1 && $0.method == "pdfkit" })
        #expect(detail.document.blocks.contains { $0.page == 2 && $0.method == "pdfkit" })
        #expect(try await engine.artifacts.data(for: #require(detail.artifacts.first)) == data)
    }

    @Test func tableChunksRepeatHeadersAndCodeSymbolsRemain() {
        let block = ContentBlock(
            kind: .table,
            markdown: "| Name | Score |\n| --- | --- |\n| Alpha | 42 |\n| Beta | 51 |",
            text: "Name Score Alpha 42 Beta 51",
            section: "Results"
        )
        let chunks = CaptureChunker.chunks(block)
        #expect(chunks.count == 2)
        #expect(chunks
            .allSatisfy { $0.contains("Name") && $0.contains("Score") && $0.contains("Results") })
        let code = ContentBlock(
            kind: .code,
            markdown: "```\nlet value_name = 42\n```",
            text: "let value_name = 42"
        )
        #expect(CaptureChunker.chunks(code).first == "let value_name = 42")
        let math = ContentBlock(
            kind: .paragraph,
            markdown: "value_name * scale",
            text: "value_name * scale",
            method: "pdfkit"
        )
        #expect(CaptureChunker.chunks(math).first == "value_name * scale")
        let merged = ContentBlock(
            kind: .table,
            markdown: "```text\nFirst\trow\nSecond\trow\n```",
            text: "First row\nSecond row"
        )
        #expect(CaptureChunker.chunks(merged).joined().contains("First row"))
    }

    @Test func decodesDeclaredAndBOMTextWithoutReplacementCharacters() throws {
        let text = "Café naïve"
        let bytes = try #require(text.data(using: .windowsCP1252))
        #expect(try IngestionEngine.decodeText(bytes, encoding: "windows-1252") == text)
        #expect(try IngestionEngine.decodeText(#require(text.data(using: .utf16))) == text)
        #expect(throws: CaptureError.self) {
            try IngestionEngine.decodeText(Data([0xFF, 0xAA, 0xAB]))
        }
    }

    @Test func retryDuringCancellationIsNotCancelledByOldWorker() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let fetcher = HeldFetcher()
        let engine = IngestionEngine(
            store: fixture.store,
            fetcher: fetcher,
            summarizer: FixtureLinkSummarizer()
        )
        let id = try await engine.enqueue("https://fixture.example/retry.txt")
        let runner = Task { try await engine.run() }
        await fetcher.waitUntilHeld()
        try await engine.cancel(id)
        try await fixture.store.retryCapture(id)
        await fetcher.release()
        try await runner.value
        #expect(try await fixture.store.captureJobs().first?.state == .ready)
        #expect(try await fixture.store.documents().count == 1)
    }

    @Test func publicationFailureRollsBackCanonicalAndDerivedRows() async throws {
        let fixture = try CaptureFixture(); defer { fixture.clean() }
        _ = try await fixture.store.open(seed: [])
        let db = try DatabaseQueue(path: fixture.store.url.path)
        try await db
            .write {
                try $0
                    .execute(
                        sql: "CREATE TRIGGER reject_capture BEFORE INSERT ON content_blocks BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
                    )
            }
        let engine = IngestionEngine(store: fixture.store, summarizer: FixtureLinkSummarizer())
        _ = try await engine
            .enqueue("A synthetic transaction failure must not leave a half-published note.")
        try await engine.run()
        #expect(try await fixture.store.captureJobs().first?.state == .failed)
        for table in [
            "notes",
            "note_revisions",
            "local_changes",
            "captures",
            "content_blocks",
            "sources",
            "passages",
            "index_jobs",
        ] {
            let count = try await db.read { try Int.fetchOne(
                $0,
                sql: "SELECT count(*) FROM \(table)"
            ) }
            #expect(count == 0)
        }
    }

    @Test func multicolumnPDFPreservesEmbeddedTextWithoutOCR() async throws {
        let data = try Self.pdfData(columns: true)
        let document = try await PDFExtractor()
            .extract(data, name: "columns.pdf", url: nil) { message in
                #expect(!message.contains("OCR"))
            }
        #expect(document.blocks.count == 2)
        #expect(document.blocks.allSatisfy { $0.method == "pdfkit" })
        #expect(document.markdown.contains("left firstbeacon"))
        #expect(document.markdown.contains("right secondbeacon"))
        #expect(document.warnings.contains { $0.contains("reading order") })
        #expect(!document.warnings.contains { $0.contains("low OCR confidence") })
    }

    /// Vision's document recognition needs the Neural Engine, which virtual machines such as
    /// GitHub's macOS runners don't have. There it finds nothing and the extractor falls back.
    @Test(.disabled(if: runningInVirtualMachine, "Vision document recognition needs real hardware"))
    func nativeVisionExtractsScannedPDFLocally() async throws {
        let data = try Self.pdfData(scanned: true)
        let document = try await PDFExtractor().extract(
            data,
            name: "scan.pdf",
            url: nil
        )
        #expect(document.blocks.contains { $0.method == "vision-document" && $0.page == 1 })
        #expect(document.blocks.contains { $0.page == 2 })
        #expect(document.markdown.lowercased().contains("local notes"))
    }

    private static func pdfData(scanned: Bool = false, columns: Bool = false) throws -> Data {
        let buffer = NSMutableData()
        let consumer = try #require(CGDataConsumer(data: buffer))
        var bounds = CGRect(x: 0, y: 0, width: 612, height: 792)
        let context = try #require(CGContext(consumer: consumer, mediaBox: &bounds, nil))
        let font = CTFontCreateWithName("Helvetica" as CFString, 12, nil)
        for token in ["firstbeacon", "secondbeacon"] {
            context.beginPDFPage(nil)
            let drawing: CGContext
            if scanned {
                drawing = try #require(CGContext(
                    data: nil,
                    width: 1224,
                    height: 1584,
                    bitsPerComponent: 8,
                    bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                ))
                drawing.scaleBy(x: 2, y: 2)
                drawing.setFillColor(CGColor(gray: 1, alpha: 1)); drawing.fill(bounds)
                drawing.setFillColor(CGColor(gray: 0, alpha: 1))
            } else {
                drawing = context
            }
            for line in 0 ..< 5 {
                let texts = columns
                    ? ["left \(token) line \(line): x + y", "right \(token) line \(line): a / b"]
                    :
                    [
                        "Synthetic \(token) page line \(line): durable local notes preserve the original source.",
                    ]
                for (column, text) in texts.enumerated() {
                    let attributed = NSAttributedString(
                        string: text,
                        attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font]
                    )
                    drawing.textPosition = CGPoint(x: 40 + column * 280, y: 720 - line * 24)
                    CTLineDraw(CTLineCreateWithAttributedString(attributed), drawing)
                }
            }
            if scanned {
                try context.draw(#require(drawing.makeImage()), in: bounds)
            }
            context.endPDFPage()
        }
        context.closePDF()
        return buffer as Data
    }
}

/// True on a virtual Mac, where hardware-backed frameworks behave differently.
private let runningInVirtualMachine: Bool = {
    var present: Int32 = 0
    var size = MemoryLayout<Int32>.size
    return sysctlbyname("kern.hv_vmm_present", &present, &size, nil, 0) == 0 && present == 1
}()
