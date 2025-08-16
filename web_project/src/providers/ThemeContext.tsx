"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';
import { COLORS, ColorTheme } from '../styles/Colors';

interface ThemeContextType {
  theme: ColorTheme;
  toggleTheme: () => void;
  setTheme: (theme: ColorTheme) => void;
  colors: typeof COLORS.light | typeof COLORS.dark;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ColorTheme>('dark');

  // 로컬 스토리지에서 테마 불러오기
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as ColorTheme;
    if (savedTheme && (savedTheme === 'light' || savedTheme === 'dark')) {
      setThemeState(savedTheme);
    } else {
      // 시스템 테마 감지
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      setThemeState(systemTheme);
    }
  }, []);

  // 테마 변경 시 로컬 스토리지에 저장
  useEffect(() => {
    localStorage.setItem('theme', theme);
    
    // HTML 클래스에 테마 적용
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
    
    // CSS 변수로 테마 컬러 적용
    const root = document.documentElement;
    if (theme === 'dark') {
      root.style.setProperty('--background-start', COLORS.dark.background.primary);
      root.style.setProperty('--background-end', COLORS.dark.background.secondary);
      root.style.setProperty('--text-primary', COLORS.text.dark.primary);
      root.style.setProperty('--text-secondary', COLORS.text.dark.secondary);
    } else {
      root.style.setProperty('--background-start', COLORS.light.background.primary);
      root.style.setProperty('--background-end', COLORS.light.background.secondary);
      root.style.setProperty('--text-primary', COLORS.text.light.primary);
      root.style.setProperty('--text-secondary', COLORS.text.light.secondary);
    }
  }, [theme]);

  const toggleTheme = () => {
    setThemeState(prev => prev === 'light' ? 'dark' : 'light');
  };

  const setTheme = (newTheme: ColorTheme) => {
    setThemeState(newTheme);
  };

  const colors = theme === 'dark' ? COLORS.dark : COLORS.light;
  const isDark = theme === 'dark';

  const value: ThemeContextType = {
    theme,
    toggleTheme,
    setTheme,
    colors,
    isDark,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
