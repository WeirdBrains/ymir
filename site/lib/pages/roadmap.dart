import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../content/roadmap.dart';
import '../site.dart';

const _labels = {'done': 'Done', 'in-progress': 'In progress', 'planned': 'Planned'};

/// Inline `code` spans and [text](https://url) links in the roadmap file's plain text.
List<Component> _inline(String s) {
  final link = RegExp(r'\[([^\]]+)\]\((https?://[^)\s]+)\)');
  final out = <Component>[];
  var at = 0;
  void plain(String t) {
    final parts = t.split('`');
    for (var i = 0; i < parts.length; i++) {
      if (parts[i].isNotEmpty) out.add(i.isOdd ? code([Component.text(parts[i])]) : Component.text(parts[i]));
    }
  }

  for (final m in link.allMatches(s)) {
    plain(s.substring(at, m.start));
    out.add(a(href: m.group(2)!, [Component.text(m.group(1)!)]));
    at = m.end;
  }
  plain(s.substring(at));
  return out;
}

/// The milestone list, generated from `roadmap.toml` (see `tool/gen_roadmap.py`).
class RoadmapPage extends StatelessComponent {
  const RoadmapPage({super.key});

  @override
  Component build(BuildContext context) {
    final done = milestones.where((m) => m.status == 'done').length;
    return Component.fragment([
      pageHero(eyebrow: 'Roadmap', title: 'Where $siteTitle is going', lede: roadmapIntro),
      section(classes: 'wrap roadmap', [
        p(classes: 'roadmap-meta', [
          Component.text('$done of ${milestones.length} milestones done · updated $roadmapUpdated · '),
          a(href: roadmapDesignUrl, [Component.text('design doc')]),
          Component.text(' · '),
          a(href: '$repoUrl/blob/main/ROADMAP.md', [Component.text('ROADMAP.md')]),
        ]),
        ol(classes: 'milestones', [
          for (final m in milestones)
            li(classes: 'milestone ${m.status}', id: m.id.toLowerCase(), [
              div(classes: 'milestone-head', [
                span(classes: 'milestone-id', [Component.text(m.id)]),
                h2([Component.text(m.title)]),
                span(classes: 'badge ${m.status}', [Component.text(_labels[m.status] ?? m.status)]),
              ]),
              p(_inline(m.summary)),
              p(classes: 'milestone-crates', [
                span(classes: 'k', [Component.text('Weirdnet crates: ')]),
                if (m.crates.isEmpty)
                  Component.text('none new')
                else
                  for (final c in m.crates)
                    if (unpublishedCrates.contains(c))
                      span(classes: 'crate-chip planned', attributes: {'title': 'planned, not published yet'}, [
                        Component.text('$c (planned)'),
                      ])
                    else
                      a(href: cratesIoUrl(c), classes: 'crate-chip', [Component.text(c)]),
              ]),
            ]),
        ]),
      ]),
    ]);
  }
}
