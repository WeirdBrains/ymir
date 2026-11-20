import 'package:jaspr/dom.dart';
import 'package:jaspr/jaspr.dart';
import 'package:jaspr_router/jaspr_router.dart';

import 'components/layout.dart';
import 'pages/docs.dart';
import 'pages/docs_develop.dart';
import 'pages/docs_guide.dart';
import 'pages/docs_ymirctl.dart';
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
          path: '/docs',
          title: 'Docs · $siteTitle',
          builder: (_, _) => const Layout(active: '/docs', child: DocsPage()),
        ),
        Route(
          path: '/docs/guides/deployment',
          title: 'Deployment guide · $siteTitle',
          builder: (_, _) => const Layout(active: '/docs', child: DeploymentGuidePage()),
        ),
        Route(
          path: '/docs/guides/ymirctl',
          title: 'ymirctl guide · $siteTitle',
          builder: (_, _) => const Layout(active: '/docs', child: YmirctlGuidePage()),
        ),
        Route(
          path: '/docs/develop',
          title: 'Developer guide · $siteTitle',
          builder: (_, _) => const Layout(active: '/docs', child: DevelopPage()),
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
