"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";
import { useTheme } from '../providers/ThemeContext';
import { COLORS } from '../styles/Colors';

export default function ThemeToggle() {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <Button
      onClick={toggleTheme}
      variant="ghost"
      size="icon"
      className="fixed top-4 right-4 z-50 rounded-full shadow-lg transition-all duration-300 hover:scale-110"
      style={{
        backgroundColor: isDark ? COLORS.dark.background.button : COLORS.light.background.primary,
        color: isDark ? COLORS.text.dark.primary : COLORS.text.light.primary,
        border: `1px solid ${isDark ? COLORS.dark.border : COLORS.light.background.tertiary}`,
      }}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}
    >
      {isDark ? (
        <Sun size={20} className="transition-transform duration-300 rotate-0" />
      ) : (
        <Moon size={20} className="transition-transform duration-300 rotate-0" />
      )}
    </Button>
  );
}

// 테마 선택 드롭다운 버전
export function ThemeSelector() {
  const { theme, setTheme, isDark } = useTheme();

  return (
    <div className="fixed top-4 right-4 z-50">
      <select
        value={theme}
        onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
        className="px-3 py-2 rounded-lg border transition-all duration-300"
        style={{
          backgroundColor: isDark ? COLORS.dark.background.button : COLORS.light.background.primary,
          color: isDark ? COLORS.text.dark.primary : COLORS.text.light.primary,
          border: `1px solid ${isDark ? COLORS.dark.border : COLORS.light.background.tertiary}`,
        }}
      >
        <option value="light">☀️ Light</option>
        <option value="dark">🌙 Dark</option>
      </select>
    </div>
  );
}
