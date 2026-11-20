import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';
import 'package:jaspr_router/jaspr_router.dart';

import 'components/layout.dart';
import 'pages/hardware.dart';
import 'pages/home.dart';
import 'pages/quickstart.dart';
import 'pages/roadmap.dart';
import 'site.dart';

/// Every page of the site. Static mode renders each route to its own HTML file.
class App extends StatelessComponent {
  const App({super.key});

  @override
  Component build(BuildContext context) {
    return Router(
      routes: [
        Route(
          path: '/',
          title: '$siteTitle: $siteTagline',
          builder: (_, _) => const Layout(active: '/', child: Home()),
        ),
        Route(
          path: '/quickstart',
          title: 'Quickstart · $siteTitle',
          builder: (_, _) => const Layout(active: '/quickstart', child: QuickstartPage()),
        ),
        Route(
          path: '/hardware',
          title: 'Hardware · $siteTitle',
          builder: (_, _) => const Layout(active: '/hardware', child: HardwarePage()),
        ),
        Route(
          path: '/roadmap',
          title: 'Roadmap · $siteTitle',
          builder: (_, _) => const Layout(active: '/roadmap', child: RoadmapPage()),
        ),
      ],
      errorBuilder: (_, _) => Layout(
        active: '',
        child: section(classes: 'page-hero wrap', [
          h1([Component.text('Not found')]),
          p(classes: 'lede', [
            Component.text('That page is not here. '),
            a(href: siteLink('/'), [Component.text('Back home')]),
          ]),
        ]),
      ),
    );
  }
}
