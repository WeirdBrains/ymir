{
  # Ymir as a Nix flake: builds ymird and ymirctl from source with the committed Cargo.lock.
  # Untested on NixOS so far (no Nix on the build machine); see docs/DESIGN.md "M2 status".
  description = "Ymir: a private cloud of LLM models on your own machines";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
    in
    {
      packages = forAll (pkgs: rec {
        ymir = pkgs.rustPlatform.buildRustPackage {
          pname = "ymir";
          inherit version;
          src = self;
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [ "-p" "ymird" "-p" "ymirctl" ];
          # The end-to-end tests start engines and bind loopback ports; run them in CI instead.
          doCheck = false;
          postInstall = ''
            install -Dm644 packaging/systemd/ymird.service $out/lib/systemd/user/ymird.service
            substituteInPlace $out/lib/systemd/user/ymird.service --replace-fail /usr/bin/ymird $out/bin/ymird
          '';
          meta = {
            description = "Private cloud of LLM models on your own machines (node daemon and CLI)";
            homepage = "https://github.com/WeirdBrains/ymir";
            license = pkgs.lib.licenses.bsd3;
            mainProgram = "ymirctl";
            platforms = systems;
          };
        };
        default = ymir;
      });
    };
}
