import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';

const _html = '''
<p>Ymir detects what a machine has (<code>ymirctl hw</code>) and works out a model budget. These are the design's
tiers.</p>
<table>
<thead><tr><th>Tier</th><th>Hardware</th><th>What you get</th></tr></thead>
<tbody>
<tr><td>Minimum (CPU)</td><td>x86_64 with AVX2, or ARM64; 8 GB RAM; 20 GB free disk</td><td>llama.cpp on CPU, a small model at Q4, a few tokens per second. Fine as a pool member for small jobs or embeddings.</td></tr>
<tr><td>Entry GPU</td><td>NVIDIA with 8 GB VRAM (Turing or newer), AMD RDNA2+ with 8 GB, or Apple Silicon with 16 GB</td><td>llama.cpp GPU offload with a small chat model plus a 7 to 9B Q4 model.</td></tr>
<tr><td>Comfortable</td><td>16 to 24 GB VRAM, or a 32 to 64 GB Mac</td><td>A 27 to 35B Q4 model plus a small stand-in and an embedding model.</td></tr>
<tr><td>vLLM</td><td>Linux, NVIDIA Ampere or newer (24 GB+), or AMD Instinct / RDNA3 on the ROCm list</td><td>vLLM as an extra backend: higher concurrency and FP8 or NVFP4 checkpoints. Linux only.</td></tr>
<tr><td>DGX Spark</td><td>GB10, 128 GB unified memory</td><td>A large-memory single-box target.</td></tr>
</tbody></table>
<h2 id="backends">Backends</h2>
<ul>
<li><strong>llama.cpp</strong>, through <code>llama-server</code>, on Linux, macOS and CPU-only boxes. Ymir starts it on a random loopback port with a per-process key, or uses a path you give it.</li>
<li><strong>vLLM</strong>, managed or wrapped, on Linux with NVIDIA. vLLM is not available on Windows or macOS.</li>
<li><strong>A stock server</strong> (vLLM, llama-server or anything OpenAI-compatible) already running on loopback, wrapped by Ymir. Unloading only detaches; Ymir never stops a server it didn't start.</li>
</ul>
<h2 id="disk">Disk and downloads</h2>
<p>Models are big. Setup shows the download size and asks before fetching, then checks size and SHA-256.</p>
<h2 id="platforms">Platforms</h2>
<p>Ymir builds for Linux (x86_64 and ARM64), macOS (Apple Silicon only; Intel Macs are not supported) and Windows; on Windows it runs as a logon task or as a Windows
service. BSD packages are planned (M9).</p>
''';

class HardwarePage extends StatelessComponent {
  const HardwarePage({super.key});

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      pageHero(
        eyebrow: 'Hardware',
        title: 'What it runs on',
        lede: 'From a CPU-only box to a DGX Spark. Add machines over Weirdnet as you get them.',
      ),
      prose(_html),
    ]);
  }
}
