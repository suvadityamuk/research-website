require "minitest/autorun"

class PerformanceRegressionTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)

  def read(path)
    File.read(File.join(ROOT, path))
  end

  def test_homepage_uses_a_small_avatar_asset
    config = read("_config.yml")
    avatar = config[/^\s*avatar\s*:\s*["']?([^"'\s]+)["']?/, 1]

    refute_nil avatar
    refute_equal "profile.png", avatar

    avatar_path = File.join(ROOT, "images", avatar)
    assert_path_exists avatar_path
    assert_operator File.size(avatar_path), :<=, 100 * 1024
  end

  def test_avatar_reserves_layout_space_and_prioritizes_the_lcp_image
    template = read("_includes/author-profile.html")

    assert_includes template, 'width="350"'
    assert_includes template, 'height="350"'
    assert_includes template, 'fetchpriority="high"'
    assert_includes template, 'decoding="async"'
  end

  def test_icon_fonts_do_not_hide_text_while_loading
    assert_includes read("_sass/vendor/font-awesome/_variables.scss"), "$fa-font-display        : swap"
    assert_includes read("assets/css/academicons.css"), "font-display: swap"
  end

  def test_secondary_text_and_visited_links_keep_readable_contrast
    theme = read("_sass/theme/_default.scss")

    assert_includes theme, "$light-gray                 : #5c646a"
    assert_includes theme, "--global-link-color-visited         : #2f7f93"
  end

  def test_secondary_icon_stylesheet_does_not_block_first_paint
    custom_head = read("_includes/head/custom.html")

    assert_includes custom_head, 'rel="preload" as="style"'
    assert_includes custom_head, "this.rel='stylesheet'"
  end

  def test_main_stylesheet_uses_the_icon_subset
    manifest = read("assets/css/main.scss")

    assert_includes manifest, '"font-awesome-subset"'
    refute_includes manifest, '"vendor/font-awesome/fontawesome"'
    refute_includes manifest, '"vendor/font-awesome/solid"'
    refute_includes manifest, '"vendor/font-awesome/brands"'
    assert_path_exists File.join(ROOT, "_sass", "_font-awesome-subset.scss")
  end

  def test_runtime_bundle_does_not_ship_legacy_jquery_plugins
    package = read("package.json")
    runtime = read("assets/js/_main.js")

    refute_includes package, "node_modules/jquery/dist/jquery.min.js"
    refute_includes package, "node_modules/fitvids/dist/fitvids.min.js"
    refute_includes package, "node_modules/magnific-popup/dist/jquery.magnific-popup.min.js"
    refute_includes package, "node_modules/jquery-smooth-scroll/jquery.smooth-scroll.min.js"
    refute_match(/\$\(|jQuery/, runtime)
    refute_match(/\$\(|jQuery|\$\.ajax/, read("_includes/comments-providers/staticman.html"))
  end

  def test_main_stylesheet_does_not_ship_legacy_popup_css
    refute_includes read("assets/css/main.scss"), '"vendor/magnific-popup/magnific-popup"'
  end

  def test_navigation_toggle_has_an_accessible_name_and_state
    masthead = read("_includes/masthead.html")

    assert_includes masthead, 'aria-label="Toggle navigation"'
    assert_includes masthead, 'aria-expanded="false"'
    assert_includes masthead, 'aria-controls="site-nav-menu"'
    assert_includes masthead, 'id="site-nav-menu"'
  end

  def test_profile_toggle_exposes_its_accessible_state
    profile = read("_includes/author-profile.html")

    assert_includes profile, 'aria-expanded="false"'
    assert_includes profile, 'aria-controls="author-links"'
    assert_includes profile, 'id="author-links"'
  end
end
