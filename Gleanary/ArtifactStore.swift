import Darwin
import Foundation

actor ArtifactStore {
    let root: URL

    init(root: URL) {
        self.root = root
    }

    func retain(
        _ data: Data,
        kind: ArtifactKind,
        name: String,
        sourceURL: URL? = nil,
        textEncoding: String? = nil
    ) throws -> Artifact {
        guard data.count <= CaptureLimits.artifactBytes else { throw CaptureError.tooLarge }
        let hash = ContentHash.bytes(data)
        let url = try fileURL(hash)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let existing = try? Data(contentsOf: url, options: .mappedIfSafe)
        if existing.map(ContentHash.bytes) != hash {
            try data.write(to: url, options: .atomic)
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.synchronize()
            try Self.syncDirectory(url.deletingLastPathComponent())
            try Self.syncDirectory(root)
            try Self.syncDirectory(root.deletingLastPathComponent())
        }
        return Artifact(
            hash: hash,
            kind: kind,
            name: String(name.prefix(240)),
            byteCount: data.count,
            sourceURL: sourceURL,
            textEncoding: textEncoding
        )
    }

    func retainFile(_ url: URL) throws -> Artifact {
        let access = url.startAccessingSecurityScopedResource()
        defer {
            if access {
                url.stopAccessingSecurityScopedResource()
            }
        }
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true else { throw CaptureError.unsupported }
        guard let size = values.fileSize,
              size <= CaptureLimits.artifactBytes else { throw CaptureError.tooLarge }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let kind = try Self.kind(data: data, extension: url.pathExtension)
        return try retain(data, kind: kind, name: url.lastPathComponent)
    }

    func data(for artifact: Artifact) throws -> Data {
        let url = try fileURL(artifact.hash)
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              data.count == artifact.byteCount,
              ContentHash.bytes(data) == artifact.hash else { throw CaptureError.missingArtifact }
        return data
    }

    func fileURL(_ hash: String) throws -> URL {
        guard hash.count == 64,
              hash.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw CaptureError.missingArtifact }
        return root.appending(path: String(hash.prefix(2))).appending(path: hash)
    }

    static func syncDirectory(_ url: URL) throws {
        let descriptor = Darwin.open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw CocoaError(.fileWriteUnknown) }
        defer { Darwin.close(descriptor) }
        guard fsync(descriptor) == 0 else { throw CocoaError(.fileWriteUnknown) }
    }

    static func kind(
        data: Data,
        extension suffix: String = "",
        mime: String? = nil
    ) throws -> ArtifactKind {
        if data.prefix(1024).range(of: Data("%PDF-".utf8)) != nil {
            return .pdf
        }
        let prefix = String(decoding: data.prefix(512), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if mime == "text/html" || ["html", "htm"].contains(suffix.lowercased()) || prefix
            .hasPrefix("<!doctype html") || prefix.hasPrefix("<html")
        {
            return .html
        }
        // Some sites negotiate Markdown for non-browser clients, at an extensionless URL.
        if ["md", "markdown"].contains(suffix.lowercased())
            || ["text/markdown", "text/x-markdown"].contains(mime)
        {
            return .markdown
        }
        if suffix.lowercased() == "txt" || mime == "text/plain" {
            return .text
        }
        if suffix
            .lowercased() == "pdf" || mime == "application/pdf"
        {
            throw CaptureError.invalidPDF
        }
        throw CaptureError.unsupported
    }
}
