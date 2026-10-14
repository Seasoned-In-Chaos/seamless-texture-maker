//! The design system: colour and shape tokens, type, and egui styling.
//!
//! The look is monochrome and precise: near-black surfaces, controls defined
//! by a one-pixel outline; one accent colour, used only for selection and slider
//! fill; a light primary button; one corner-radius scale. Widgets read
//! [`TOKENS`] instead of hard-coding values.

use std::sync::Arc;

use egui::epaint::text::{HintingTarget, SmoothHinting};
use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, FontTweak, Stroke,
    TextStyle, Visuals,
};

// Geist and Geist Mono, SIL Open Font License 1.1 (see assets/fonts/OFL.txt).
const GEIST_REGULAR: &[u8] = include_bytes!("../assets/fonts/Geist-Regular.ttf");
const GEIST_MEDIUM: &[u8] = include_bytes!("../assets/fonts/Geist-Medium.ttf");
const GEIST_SEMIBOLD: &[u8] = include_bytes!("../assets/fonts/Geist-SemiBold.ttf");
const GEIST_MONO: &[u8] = include_bytes!("../assets/fonts/GeistMono-Regular.ttf");
// Lucide icons, ISC license (see assets/fonts/Lucide-LICENSE.txt).
const LUCIDE: &[u8] = include_bytes!("../assets/fonts/Lucide.ttf");

const MEDIUM_FAMILY: &str = "medium";
const SEMIBOLD_FAMILY: &str = "semibold";
const ICON_FAMILY: &str = "icons";

pub const BODY_SIZE: f32 = 13.0;
/// Whole pixel sizes only: a fractional size cannot be fitted to the grid.
pub const SMALL_SIZE: f32 = 12.0;
pub const TITLE_SIZE: f32 = 18.0;
const MONO_SIZE: f32 = 12.0;

pub const TOP_BAR_HEIGHT: f32 = 44.0;
pub const STATUS_HEIGHT: f32 = 26.0;
pub const INSPECTOR_WIDTH: f32 = 320.0;
pub const CHANNEL_STRIP_HEIGHT: f32 = 96.0;
pub const PANEL_PADDING: i8 = 16;
pub const CONTROL_HEIGHT: f32 = 28.0;
/// How long hover and selection changes take to settle, in seconds.
pub const MOTION: f32 = 0.14;

/// Colour and shape tokens.
pub struct Tokens {
    /// Area behind the image: the darkest tone.
    pub canvas: Color32,
    /// Bars and panels.
    pub surface: Color32,
    /// Menus, dialogs and floating bars.
    pub popup: Color32,
    /// Controls at rest.
    pub raised: Color32,
    pub raised_hover: Color32,
    /// Pressed controls and the selected segment.
    pub raised_active: Color32,
    /// The only line colour: a faint divider.
    pub hairline: Color32,
    /// The one-pixel edge that defines a control against its surface.
    pub outline: Color32,
    /// A faint light edge: the top of lifted controls, the rim of an image.
    pub sheen: Color32,
    pub text: Color32,
    /// Secondary text. At least 4.5:1 against every surface it sits on.
    pub text_dim: Color32,
    /// The one accent: slider fill, the selected thumbnail, focus.
    pub accent: Color32,
    pub accent_soft: Color32,
    /// The primary action is light on dark, not coloured.
    pub primary: Color32,
    pub primary_hover: Color32,
    pub primary_text: Color32,
    pub danger: Color32,
    pub shadow: Color32,
    pub scrim: Color32,
    pub radius_sm: u8,
    pub radius: u8,
    pub radius_lg: u8,
}

/// The single dark theme.
pub const TOKENS: Tokens = Tokens {
    canvas: Color32::from_rgb(13, 13, 15),
    surface: Color32::from_rgb(22, 22, 24),
    popup: Color32::from_rgb(28, 28, 31),
    raised: Color32::from_rgb(33, 33, 37),
    raised_hover: Color32::from_rgb(43, 43, 48),
    raised_active: Color32::from_rgb(56, 56, 62),
    hairline: Color32::from_rgb(44, 44, 49),
    outline: Color32::from_rgb(60, 60, 66),
    // White at 7% opacity, premultiplied.
    sheen: Color32::from_rgba_premultiplied(18, 18, 18, 18),
    text: Color32::from_rgb(246, 246, 248),
    text_dim: Color32::from_rgb(164, 164, 172),
    accent: Color32::from_rgb(96, 156, 250),
    accent_soft: Color32::from_rgb(34, 52, 86),
    primary: Color32::from_rgb(236, 236, 240),
    primary_hover: Color32::WHITE,
    primary_text: Color32::from_rgb(16, 16, 18),
    danger: Color32::from_rgb(242, 112, 112),
    shadow: Color32::from_black_alpha(150),
    scrim: Color32::from_black_alpha(150),
    radius_sm: 3,
    radius: 4,
    radius_lg: 6,
};

// One ascending corner-radius scale, checked at compile time.
const _: () = assert!(TOKENS.radius_sm < TOKENS.radius && TOKENS.radius < TOKENS.radius_lg);

pub fn medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(MEDIUM_FAMILY.into()))
}

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD_FAMILY.into()))
}

/// The icon font; its glyphs are listed in [`crate::icons`].
pub fn icon_font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(ICON_FAMILY.into()))
}

/// Blends two colours: `from` at 0, `to` at 1.
pub fn mix(from: Color32, to: Color32, t: f32) -> Color32 {
    from.lerp_to_gamma(to, t.clamp(0.0, 1.0))
}

/// Glyphs fitted to the pixel grid in both directions and drawn at whole
/// pixel positions. At 100% display scale this is the difference between
/// stems that land on a pixel and stems smeared across two.
fn crisp_text() -> FontTweak {
    let fitted =
        SmoothHinting { light: false, symmetric_rendering: false, preserve_linear_metrics: false };
    FontTweak {
        hinting: Some(true),
        hinting_target: HintingTarget::Smooth(fitted),
        subpixel_binning: Some(false),
        ..FontTweak::default()
    }
}

/// Installs Geist for text and Geist Mono for numbers, with egui's bundled
/// fonts kept behind them for glyphs Geist does not have.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let embedded = [
        ("geist-regular", GEIST_REGULAR),
        ("geist-medium", GEIST_MEDIUM),
        ("geist-semibold", GEIST_SEMIBOLD),
        ("geist-mono", GEIST_MONO),
    ];
    for (name, bytes) in embedded {
        let data = FontData::from_static(bytes).tweak(crisp_text());
        fonts.font_data.insert(name.to_owned(), Arc::new(data));
    }
    let fallback = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let stack = |primary: &str| {
        let mut names = vec![primary.to_owned()];
        names.extend(fallback.iter().cloned());
        names
    };
    fonts.families.insert(FontFamily::Proportional, stack("geist-regular"));
    fonts.families.insert(FontFamily::Name(MEDIUM_FAMILY.into()), stack("geist-medium"));
    fonts.families.insert(FontFamily::Name(SEMIBOLD_FAMILY.into()), stack("geist-semibold"));
    fonts.families.entry(FontFamily::Monospace).or_default().insert(0, "geist-mono".to_owned());
    // Icons are outlines, not text: no grid fitting, which would bend them.
    fonts.font_data.insert("lucide".to_owned(), Arc::new(FontData::from_static(LUCIDE)));
    fonts.families.insert(FontFamily::Name(ICON_FAMILY.into()), vec!["lucide".to_owned()]);
    ctx.set_fonts(fonts);
}

fn visuals() -> Visuals {
    let t = &TOKENS;
    let mut v = Visuals::dark();
    v.panel_fill = t.surface;
    v.window_fill = t.popup;
    v.extreme_bg_color = t.raised;
    v.faint_bg_color = t.raised;
    v.override_text_color = Some(t.text);
    v.hyperlink_color = t.accent;
    v.error_fg_color = t.danger;
    v.window_corner_radius = CornerRadius::same(t.radius_lg);
    v.menu_corner_radius = CornerRadius::same(t.radius);
    v.window_stroke = Stroke::new(1.0, t.outline);
    v.window_shadow = egui::Shadow { offset: [0, 8], blur: 20, spread: 0, color: t.shadow };
    v.popup_shadow = egui::Shadow { offset: [0, 4], blur: 12, spread: 0, color: t.shadow };
    // Whole-pixel glyph positions for the bundled fallback fonts as well.
    v.text_options.subpixel_binning = false;
    v.selection.bg_fill = t.accent_soft;
    v.selection.stroke = Stroke::new(1.0, t.accent);
    style_widgets(&mut v);
    v
}

/// Controls are a tone on their surface, defined by a one-pixel outline.
fn style_widgets(v: &mut Visuals) {
    let t = &TOKENS;
    let radius = CornerRadius::same(t.radius_sm);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = t.surface;
    w.noninteractive.weak_bg_fill = t.surface;
    w.noninteractive.bg_stroke = Stroke::new(1.0, t.hairline);
    w.noninteractive.fg_stroke = Stroke::new(1.0, t.text_dim);
    w.noninteractive.corner_radius = radius;
    let states = [
        (&mut w.inactive, t.raised),
        (&mut w.hovered, t.raised_hover),
        (&mut w.active, t.raised_active),
        (&mut w.open, t.raised_hover),
    ];
    for (state, fill) in states {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = Stroke::new(1.0, t.outline);
        state.fg_stroke = Stroke::new(1.0, t.text);
        state.corner_radius = radius;
        state.expansion = 0.0;
    }
}

/// Installs the theme: fonts, colours, type scale and spacing.
pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    // There is one theme. Pinning it keeps the styling below in force when the
    // system switches between light and dark.
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.set_visuals(visuals());
    ctx.global_style_mut(|style| {
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(SMALL_SIZE)),
            (TextStyle::Body, FontId::proportional(BODY_SIZE)),
            (TextStyle::Button, FontId::proportional(BODY_SIZE)),
            (TextStyle::Heading, semibold(BODY_SIZE)),
            (TextStyle::Monospace, FontId::monospace(MONO_SIZE)),
        ]
        .into();
        // Numbers are set in the mono face so digits do not jitter as they change.
        style.drag_value_text_style = TextStyle::Monospace;
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        // Every gap and padding is a step of the 4 px spacing scale.
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.interact_size = egui::vec2(CONTROL_HEIGHT, CONTROL_HEIGHT);
        style.spacing.combo_width = 132.0;
        style.spacing.menu_margin = egui::Margin::same(8);
        style.spacing.window_margin = egui::Margin::same(24);
        style.spacing.scroll = egui::style::ScrollStyle::thin();
        style.visuals.indent_has_left_vline = false;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luminance(color: Color32) -> f32 {
        let channel = |c: u8| {
            let v = f32::from(c) / 255.0;
            if v <= 0.03928 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
    }

    fn contrast(a: Color32, b: Color32) -> f32 {
        let (hi, lo) = (luminance(a).max(luminance(b)), luminance(a).min(luminance(b)));
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn body_and_secondary_text_meet_wcag_aa_on_every_surface() {
        let surfaces = [TOKENS.canvas, TOKENS.surface, TOKENS.popup, TOKENS.raised];
        for surface in surfaces {
            assert!(contrast(TOKENS.text, surface) >= 4.5);
            assert!(contrast(TOKENS.text_dim, surface) >= 4.5, "dim text on {surface:?}");
        }
    }

    #[test]
    fn text_on_hovered_and_selected_controls_meets_wcag_aa() {
        assert!(contrast(TOKENS.text, TOKENS.raised_hover) >= 4.5);
        assert!(contrast(TOKENS.text, TOKENS.raised_active) >= 4.5);
    }

    #[test]
    fn primary_button_text_meets_wcag_aa() {
        assert!(contrast(TOKENS.primary_text, TOKENS.primary) >= 4.5);
        assert!(contrast(TOKENS.primary_text, TOKENS.primary_hover) >= 4.5);
    }

    #[test]
    fn accent_and_error_colours_are_legible_on_surfaces() {
        assert!(contrast(TOKENS.accent, TOKENS.surface) >= 4.5);
        assert!(contrast(TOKENS.danger, TOKENS.surface) >= 4.5);
    }

    #[test]
    fn mixing_runs_from_the_first_colour_to_the_second_and_clamps() {
        let (from, to) = (Color32::from_rgb(0, 0, 0), Color32::from_rgb(200, 100, 50));

        assert_eq!(mix(from, to, 0.0), from);
        assert_eq!(mix(from, to, 1.0), to);
        assert_eq!(mix(from, to, 0.5), Color32::from_rgb(100, 50, 25));
        assert_eq!(mix(from, to, 7.0), to);
    }

    #[test]
    fn surfaces_step_up_in_tone_from_canvas_to_active() {
        let tones = [
            TOKENS.canvas,
            TOKENS.surface,
            TOKENS.popup,
            TOKENS.raised,
            TOKENS.raised_hover,
            TOKENS.raised_active,
        ];

        assert!(tones.windows(2).all(|pair| luminance(pair[0]) < luminance(pair[1])));
    }
}
