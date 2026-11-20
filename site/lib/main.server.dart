/// The entrypoint for the **server** environment (static pre-rendering).
library;

import 'package:jaspr/dom.dart';
import 'package:jaspr/server.dart';

import 'app.dart';
import 'main.server.options.dart';
import 'site.dart';

void main() {
  Jaspr.initializeApp(options: defaultServerOptions);

  runApp(
    Document(
      title: siteTitle,
      lang: 'en',
      base: siteBase,
      meta: const {'description': siteDescription},
      head: [
        meta(attributes: {'property': 'og:type'}, content: 'website'),
        meta(attributes: {'property': 'og:site_name'}, content: siteTitle),
        meta(attributes: {'property': 'og:description'}, content: siteDescription),
        meta(attributes: {'property': 'og:image'}, content: '$siteUrl/images/social.png'),
        meta(name: 'twitter:card', content: 'summary_large_image'),
        meta(name: 'theme-color', content: '#1A1D23'),
        link(
          href: siteLink('fonts/InterVariable.woff2'),
          rel: 'preload',
          attributes: {'as': 'font', 'type': 'font/woff2', 'crossorigin': ''},
        ),
        link(href: siteLink('styles.css'), rel: 'stylesheet'),
        link(href: siteLink('favicon.svg'), rel: 'icon', type: 'image/svg+xml'),
        link(href: siteLink('favicon.ico'), rel: 'icon', attributes: {'sizes': '48x48'}),
        link(href: siteLink('images/apple-touch-icon.png'), rel: 'apple-touch-icon'),
      ],
      body: const App(),
    ),
  );
}
