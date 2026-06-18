import { NavLink, Outlet } from 'react-router-dom';
import { Sun, Moon, Zap, Users, Camera, BookOpen, Settings } from 'lucide-react';
import { useApp } from '../lib/context';
import ToastContainer from './ToastContainer';

const NAV = [
  { to: '/',           label: 'Inference',   icon: Zap      },
  { to: '/team',       label: 'Équipe',      icon: Users    },
  { to: '/contribute', label: 'Contribuer',  icon: Camera   },
  { to: '/vocabulary', label: 'Vocabulaire', icon: BookOpen },
  { to: '/admin',      label: 'Admin',       icon: Settings },
];

export default function Layout() {
  const { theme, toggleTheme } = useApp();

  return (
    <div className={`${theme} flex h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50`}>

      {/* ── Left sidebar ─────────────────────────────── */}
      <aside className="shrink-0 hidden md:flex flex-col w-14 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 z-40">

        {/* Logo */}
        <div className="shrink-0 h-14 flex items-center justify-center border-b border-zinc-200 dark:border-zinc-800">
          <div className="w-8 h-8 bg-zinc-900 dark:bg-zinc-50 rounded-lg flex items-center justify-center">
            <span className="text-[11px] font-bold text-white dark:text-zinc-900 tracking-tight">SI</span>
          </div>
        </div>

        {/* Nav icons */}
        <nav className="flex-1 flex flex-col items-center py-3 gap-0.5">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `group relative w-10 h-10 rounded-xl flex items-center justify-center transition-colors duration-150 cursor-pointer ${
                  isActive
                    ? 'bg-zinc-900 dark:bg-zinc-50 text-white dark:text-zinc-900'
                    : 'text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50 hover:bg-zinc-100 dark:hover:bg-zinc-800'
                }`
              }
            >
              <Icon className="w-4.5 h-4.5" />

              {/* Tooltip */}
              <div className="absolute left-full ml-3 z-50 pointer-events-none">
                <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex items-center gap-0">
                  {/* Arrow */}
                  <div className="w-0 h-0 border-t-4 border-b-4 border-r-4 border-transparent border-r-zinc-900 dark:border-r-zinc-700" />
                  <span className="px-2.5 py-1.5 text-xs font-medium bg-zinc-900 dark:bg-zinc-700 text-white rounded-lg whitespace-nowrap shadow-lg">
                    {label}
                  </span>
                </div>
              </div>
            </NavLink>
          ))}
        </nav>

        {/* Theme toggle */}
        <div className="shrink-0 flex items-center justify-center p-2 border-t border-zinc-200 dark:border-zinc-800">
          <button
            onClick={toggleTheme}
            className="group relative w-10 h-10 rounded-xl flex items-center justify-center text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
            title="Changer le thème"
          >
            <Sun  className="w-4.5 h-4.5 hidden dark:block" />
            <Moon className="w-4.5 h-4.5 block dark:hidden" />
          </button>
        </div>
      </aside>

      {/* ── Page content ─────────────────────────────── */}
      <main className="flex-1 min-h-0 overflow-hidden">
        <Outlet />
      </main>

      {/* ── Mobile bottom nav ────────────────────────── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 h-16 border-t border-zinc-200 dark:border-zinc-800 bg-white/90 dark:bg-zinc-950/90 backdrop-blur-md flex items-center justify-around px-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex flex-col items-center gap-1 px-3 py-2 rounded-xl transition-colors ${
                isActive
                  ? 'text-zinc-900 dark:text-zinc-50'
                  : 'text-zinc-400'
              }`
            }
          >
            <Icon className="w-5 h-5" />
            <span className="text-[10px] font-medium">{label}</span>
          </NavLink>
        ))}
        <button
          onClick={toggleTheme}
          className="flex flex-col items-center gap-1 px-3 py-2 rounded-xl text-zinc-400 cursor-pointer"
        >
          <Sun  className="w-5 h-5 hidden dark:block" />
          <Moon className="w-5 h-5 block dark:hidden" />
          <span className="text-[10px] font-medium">Thème</span>
        </button>
      </nav>

      <ToastContainer />
    </div>
  );
}
