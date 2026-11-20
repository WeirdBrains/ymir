import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';

import '../site.dart';

const _nav = [
  ('/', 'Overview'),
  ('/quickstart', 'Quickstart'),
  ('/hardware', 'Hardware'),
  ('/roadmap', 'Roadmap'),
];

/// Header, main and footer around a page.
class Layout extends StatelessComponent {
  const Layout({required this.active, required this.child, super.key});

  final String active;
  final Component child;

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      a(href: '#content', classes: 'skip-link', [Component.text('Skip to content')]),
      header(classes: 'site-header', [
        div(classes: 'header-inner wrap', [
          a(href: siteLink('/'), classes: 'logo', [img(src: siteLink('images/ymir-lockup.svg'), alt: siteTitle)]),
          nav(classes: 'site-nav', [
            for (final (href, label) in _nav)
              a(href: siteLink(href), classes: active == href ? 'active' : null, [Component.text(label)]),
            a(href: bragiSiteUrl, [Component.text('Bragi')]),
            a(href: repoUrl, target: Target.blank, attributes: {'rel': 'noopener noreferrer'}, [Component.text('GitHub')]),
          ]),
        ]),
      ]),
      Component.element(tag: 'main', id: 'content', children: [child]),
      footer(classes: 'site-footer', [
        div(classes: 'footer-inner wrap', [
          p([
            Component.text('Ymir is a '),
            a(href: mainSiteUrl, [Component.text('Weird Brains')]),
            Component.text(' project, open source under the BSD-3-Clause license.'),
          ]),
          div(classes: 'footer-links', [
            a(href: repoUrl, [Component.text('GitHub')]),
            a(href: bragiSiteUrl, [Component.text('Bragi, the chat app')]),
            a(href: weirdnetUrl, [Component.text('Weirdnet')]),
            a(href: siteLink('/roadmap'), [Component.text('Roadmap')]),
          ]),
        ]),
      ]),
    ]);
  }
}

/// A page title block.
Component pageHero({required String eyebrow, required String title, required String lede}) {
  return section(classes: 'page-hero wrap', [
    span(classes: 'eyebrow', [Component.text(eyebrow)]),
    h1([Component.text(title)]),
    p(classes: 'lede', [Component.text(lede)]),
  ]);
}

/// A prose body written as HTML.
Component prose(String html) {
  return div(classes: 'doc wrap', [
    article(classes: 'prose', [RawText(keepPre(html))]),
  ]);
}

/// The static renderer indents every line of raw HTML, which shows inside `<pre>`; line
/// breaks there become `&#10;` so code blocks keep their own indentation.
String keepPre(String html) => html.replaceAllMapped(
  RegExp(r'<pre>[\s\S]*?</pre>'),
  (m) => m[0]!.replaceAll('\n', '&#10;'),
);
