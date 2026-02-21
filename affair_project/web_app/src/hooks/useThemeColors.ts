import { useTheme } from '../providers/ThemeContext';
import { COLORS } from '../styles/Colors';

export function useThemeColors() {
  const { theme, isDark } = useTheme();

  // 현재 테마의 컬러
  const currentColors = isDark ? COLORS.dark : COLORS.light;
  const currentTextColors = isDark ? COLORS.text.dark : COLORS.text.light;

  // 테마별 컬러 객체
  const themeColors = {
    // 배경
    background: {
      primary: currentColors.background.primary,
      secondary: currentColors.background.secondary,
      tertiary: isDark ? COLORS.dark.background.card : COLORS.light.background.tertiary,
      card: isDark ? COLORS.dark.background.card : COLORS.white,
      cardLight: isDark ? COLORS.dark.background.cardLight : COLORS.white,
      icon: isDark ? COLORS.dark.background.icon : COLORS.light.background.primary,
      button: isDark ? COLORS.dark.background.button : COLORS.primary,
      buttonHover: isDark ? COLORS.dark.background.buttonHover : COLORS.secondary,
    },

    // 텍스트
    text: {
      primary: currentTextColors.primary,
      secondary: currentTextColors.secondary,
      tertiary: currentTextColors.tertiary,
      quaternary: currentTextColors.quaternary,
      quinary: currentTextColors.quinary,
    },

    // 테두리
    border: isDark ? COLORS.dark.border : COLORS.light.background.tertiary,

    // 차트
    chart: {
      grid: isDark ? COLORS.dark.grid : COLORS.light.background.tertiary,
      axis: isDark ? COLORS.dark.axis : currentTextColors.quinary,
      tooltip: {
        background: COLORS.tooltip.background,
        border: COLORS.tooltip.border,
        text: COLORS.tooltip.text,
      },
    },

    // 브랜드 컬러 (테마와 무관하게 고정)
    brand: {
      primary: COLORS.primary,
      secondary: COLORS.secondary,
      success: COLORS.success,
      warning: COLORS.warning,
      danger: COLORS.danger,
      teal: COLORS.teal,
      rose: COLORS.rose,
      violet: COLORS.violet,
    },

    // 차트 컬러 팔레트
    chartPalette: COLORS.chart,
  };

  return {
    theme,
    isDark,
    colors: themeColors,
    // 편의 함수들
    getBackground: (variant: 'primary' | 'secondary' | 'tertiary' | 'card' | 'cardLight' | 'icon' | 'button' | 'buttonHover') => 
      themeColors.background[variant],
    getText: (variant: 'primary' | 'secondary' | 'tertiary' | 'quaternary' | 'quinary') => 
      themeColors.text[variant],
    getBrand: (variant: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'teal' | 'rose' | 'violet') => 
      themeColors.brand[variant],
    getChartColor: (index: number) => COLORS.chart[Object.keys(COLORS.chart)[index] as keyof typeof COLORS.chart],
  };
}
