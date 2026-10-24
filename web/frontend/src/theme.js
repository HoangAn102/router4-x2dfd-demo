// Light/dark theme helpers.
//
// The inline script in index.html applies the saved theme before React
// renders (no flash). Keep its storage key and default in sync with this file.

const STORAGE_KEY = 'theme';
const DEFAULT_THEME = 'light';
const TRANSITION_CLASS = 'theme-transition';
const TRANSITION_MS = 300;

let transitionTimer = null;

function isTheme(value) {
  return value === 'light' || value === 'dark';
}

export function getInitialTheme() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (isTheme(saved)) return saved;
  } catch {}
  return DEFAULT_THEME;
}

export function applyTheme(theme) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
}

export function toggleTheme() {
  const root = document.documentElement;
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';

  // Enable color transitions only while switching, so normal hover/transform
  // transitions are untouched. CSS skips this under prefers-reduced-motion.
  root.classList.add(TRANSITION_CLASS);
  window.clearTimeout(transitionTimer);
  transitionTimer = window.setTimeout(() => {
    root.classList.remove(TRANSITION_CLASS);
  }, TRANSITION_MS);

  applyTheme(next);
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {}
  return next;
}
