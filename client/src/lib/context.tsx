import { createContext, useContext, type ReactNode } from 'react';
import { useTheme } from '../hooks/useTheme';
import { useToast } from '../hooks/useToast';
import type { Theme, Toast } from './types';

interface AppContextType {
  theme: Theme;
  toggleTheme: () => void;
  toasts: Toast[];
  addToast: (type: Toast['type'], message: string, duration?: number) => void;
  removeToast: (id: string) => void;
}

const AppContext = createContext<AppContextType | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const { theme, toggleTheme } = useTheme();
  const { toasts, addToast, removeToast } = useToast();

  return (
    <AppContext.Provider value={{ theme, toggleTheme, toasts, addToast, removeToast }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
