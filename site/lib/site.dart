/// Public URLs and copy used across the Ymir site.
library;

/// Where the site is served. Project pages live under `/ymir/`; a custom domain would be `/`.
/// Override at build time: `jaspr build --dart-define=SITE_BASE=/`.
const siteBase = String.fromEnvironment('SITE_BASE', defaultValue: '/ymir/');
const siteUrl = String.fromEnvironment('SITE_URL', defaultValue: 'https://weirdbrains.github.io/ymir');

/// A site-internal link that works under [siteBase].
String siteLink(String path) => '$siteBase${path.startsWith('/') ? path.substring(1) : path}';

const siteTitle = 'Ymir';
const siteTagline = 'Your own cloud of LLM models';
const siteDescription =
    'Ymir turns your own computers into one private cloud of LLM models, shared over Weirdnet. '
    'No hosted service, no open ports, no API keys. Open source, BSD-3-Clause.';

const repoUrl = 'https://github.com/WeirdBrains/ymir';

/// The latest release; bump with each release.
const latestVersion = '0.3.1';
const releaseUrl = '$repoUrl/releases/tag/v$latestVersion';
const taurUrl = 'https://pkg.taur.weirdbrains.com';

/// Installs ymir-bin from TAUR with yay, leaving yay's default AUR alone (TAUR docs/yay.md).
const taurInstall = "yay --aururl $taurUrl --aurrpcurl '$taurUrl/rpc?' -S ymir-bin";
const bragiRepoUrl = 'https://github.com/WeirdBrains/bragi';
const orgUrl = 'https://github.com/WeirdBrains';
const weirdnetUrl = 'https://net.weirdbrains.com';
const mainSiteUrl = 'https://weirdbrains.com';
// Sibling site. Until both custom domains are live this is its GitHub Pages address; the Pages
// workflow passes SIBLING_URL=https://bragi.weirdbrains.com/ when the CUSTOM_DOMAIN variable is set.
const bragiSiteUrl = String.fromEnvironment('SIBLING_URL', defaultValue: 'https://weirdbrains.github.io/bragi/');
String cratesIoUrl(String crate) => 'https://crates.io/crates/$crate';
