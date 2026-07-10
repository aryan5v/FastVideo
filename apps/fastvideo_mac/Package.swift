// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "FastVideoMac",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "FastVideoMac", targets: ["FastVideoMac"]),
    ],
    targets: [
        .executableTarget(
            name: "FastVideoMac",
            path: "Sources/FastVideoMac"
        ),
    ],
    swiftLanguageModes: [.v5]
)
