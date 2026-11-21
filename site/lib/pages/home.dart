import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../content/roadmap.dart';
import '../site.dart';

Component _card(String title, Object text) => div(classes: 'card', [
  h3([Component.text(title)]),
  p([text is Component ? text : Component.text(text as String)]),
]);

class Home extends StatelessComponent {
  const Home({super.key});

  @override
  Component build(BuildContext context) {
    final current = milestones.where((m) => m.status == 'in-progress').map((m) => '${m.id} ${m.title}').join(', ');
    final done = milestones.where((m) => m.status == 'done').map((m) => m.id).join(', ');
    return Component.fragment([
      section(classes: 'hero wrap', [
        img(src: siteLink('images/ymir-mark.svg'), alt: '', classes: 'hero-mark'),
        span(classes: 'eyebrow', [Component.text('Self-hosted · Open source · Over Weirdnet')]),
        h1([Component.text('Your own cloud of LLM models')]),
        p(classes: 'lede', [
          Component.text(
            'Ymir turns a computer with a decent GPU, or just a CPU, into part of a private model cloud with one '
            'command. Start with one machine and add more. The pool acts as one endpoint, and the models are spread '
            'across your own devices over Weirdnet: no open HTTPS port, no reverse proxy, no API keys.',
          ),
        ]),
        div(classes: 'actions', [
          a(href: siteLink('/quickstart'), classes: 'button solid', [Component.text('Quickstart')]),
          a(href: siteLink('/roadmap'), classes: 'button', [Component.text('Roadmap')]),
          a(href: repoUrl, classes: 'button', [Component.text('GitHub')]),
        ]),
      ]),
      section(classes: 'section band', [
        div(classes: 'wrap', [
          div(classes: 'section-head', [
            h2([Component.text('Three roles, placed freely')]),
            p([
              Component.text(
                'Run all three on one box, or spread them over any number of devices. They speak the same '
                'protocols whether a peer is local or remote.',
              ),
            ]),
          ]),
          div(classes: 'grid', [
            _card(
              'Model nodes',
              'Run the models: llama.cpp everywhere, vLLM where it is supported, or a stock server Ymir wraps. They '
                  'advertise what they have and load or unload on command.',
            ),
            _card(
              'Inference nodes',
              'Coordinate the pool: one model catalog, routing, a stand-in model while a big one loads, and failover. '
                  'Load balancing and synced context are built; cloud-wide swapping is planned.',
            ),
            _card(
              'App nodes',
              'Ask for models and send prompts: Bragi, ymirctl, or the loopback-only OpenAI-compatible bridge for Jan '
                  'and scripts on 127.0.0.1.',
            ),
          ]),
        ]),
      ]),
      section(classes: 'section', [
        div(classes: 'wrap', [
          div(classes: 'section-head', [
            h2([Component.text('What works today')]),
            p([Component.text('Status from the design doc, $roadmapUpdated.')]),
          ]),
          ul(classes: 'checks', [
            li([Component.text('One box runs all three roles in one ymird, with llama.cpp and vLLM (managed, or wrapping a server that is already running).')]),
            li([Component.text('An OpenAI-compatible endpoint on loopback: chat, completions and embeddings, so Jan and scripts work.')]),
            li([Component.text('A small stand-in model answers while a big one loads, clearly marked; models unload when idle and restart with backoff after a crash.')]),
            li([Component.text('Roles talk across devices over Weirdnet, and only devices in your own roster are let in.')]),
            li([Component.text('Milestones done: $done. In progress: $current.')]),
          ]),
          p([
            Component.text('Linux packages (tar.gz, zip, .deb and a TAUR recipe) build with the release workflow. '),
            a(href: siteLink('/roadmap'), [Component.text('See the full roadmap.')]),
          ]),
        ]),
      ]),
      section(classes: 'section band', [
        div(classes: 'wrap', [
          div(classes: 'section-head', [
            h2([
              Component.text('Part of the '),
              a(href: mainSiteUrl, [Component.text('WeirdBrains')]),
              Component.text(' family'),
            ]),
            p([Component.text('Ymir holds only the Ymir-specific app. The generic parts are published Weirdnet crates.')]),
          ]),
          div(classes: 'grid', [
            _card('Bragi', 'The chat client for desktop and phone. It finds your Ymir nodes and streams answers from your own machines.'),
            _card(
              'Weirdnet',
              Component.fragment([
                Component.text('The peer-to-peer network under every '),
                a(href: mainSiteUrl, [Component.text('WeirdBrains')]),
                Component.text(' app: identities, devices, relays and the inference protocol.'),
              ]),
            ),
            _card('Open source', 'BSD-3-Clause. Source and issues on GitHub.'),
          ]),
          div(classes: 'actions', [
            a(href: bragiSiteUrl, classes: 'button', [Component.text('Bragi')]),
            a(href: weirdnetUrl, classes: 'button', [Component.text('Weirdnet')]),
            a(href: repoUrl, classes: 'button', [Component.text('Source')]),
          ]),
        ]),
      ]),
    ]);
  }
}
