# Ymir site

Jaspr in static mode, deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`
(same pattern as `WeirdBrains/weirdnet-site`). The roadmap page is generated from `../roadmap.toml`:
`python3 tool/gen_roadmap.py` (CI runs it with `--check`).

```sh
dart pub global activate jaspr_cli 0.24.0-beta.1
dart pub get
jaspr serve                                   # local
jaspr build --sitemap-domain https://weirdbrains.github.io/ymir   # github.io build, base /ymir/
```

## Hosting and the custom domain

The site is live at [ymir.weirdbrains.com](https://ymir.weirdbrains.com/). The repository variable `CUSTOM_DOMAIN=ymir.weirdbrains.com` makes the Pages workflow build with
`SITE_BASE=/`, `SITE_URL=https://ymir.weirdbrains.com`, `SIBLING_URL=https://bragi.weirdbrains.com/` and write a `CNAME` file.
Without the variable the build is the [github.io project page](https://weirdbrains.github.io/ymir/) (base `/ymir/`).

## Checks

CI runs `python3 tool/gen_roadmap.py --check` (the roadmap pages match `roadmap.toml`) and `python3 tool/check_links.py`
(every reference to another document or URL in the page sources and READMEs is a link).
