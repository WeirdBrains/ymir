# Homebrew formula template for the WeirdBrains tap. The release workflow fills in the version and
# sha256 of the macOS tarballs. Not built or tested yet (no macOS build in M2).
class Ymir < Formula
  desc "Private cloud of LLM models on your own machines (node daemon and CLI)"
  homepage "https://github.com/WeirdBrains/ymir"
  version "0.1.0"
  license "BSD-3-Clause"

  on_arm do
    url "https://github.com/WeirdBrains/ymir/releases/download/v#{version}/ymir-#{version}-aarch64-apple-darwin.tar.gz"
    sha256 "REPLACE_WITH_AARCH64_SHA256"
  end
  on_intel do
    url "https://github.com/WeirdBrains/ymir/releases/download/v#{version}/ymir-#{version}-x86_64-apple-darwin.tar.gz"
    sha256 "REPLACE_WITH_X86_64_SHA256"
  end

  def install
    bin.install "bin/ymird", "bin/ymirctl"
    doc.install "share/doc/ymir/README.md"
    pkgshare.install "share/icons/hicolor/scalable/apps/ymir.svg"
  end

  # A launchd agent, user level: `brew services start ymir`.
  service do
    run [opt_bin/"ymird"]
    keep_alive successful_exit: false
    log_path var/"log/ymird.log"
    error_log_path var/"log/ymird.log"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/ymirctl --version")
  end
end
