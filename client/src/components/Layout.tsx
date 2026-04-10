import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Hand, BookOpen, Users, Settings, Sun, Moon, UserPlus } from 'lucide-react';
import { useApp } from '../lib/context';
import ToastContainer from './ToastContainer';

const navItems = [
  { to: '/', icon: Hand, label: 'Inference' },
  { to: '/team', icon: UserPlus, label: 'Mon equipe' },
  { to: '/contribute', icon: Users, label: 'Contribuer' },
  { to: '/vocabulary', icon: BookOpen, label: 'Vocabulaire' },
  { to: '/admin', icon: Settings, label: 'Admin' },
];

export default function Layout() {
  const { theme, toggleTheme } = useApp();
  const location = useLocation();

  return (
    <div className={`${theme} flex h-screen flex-col md:flex-row bg-white dark:bg-[#0d1117]`}>
      {/* Desktop Sidebar */}
      <nav className="hidden md:flex w-60 flex-col shrink-0 bg-white dark:bg-[#010409] border-r border-[#d0d7de] dark:border-[#30363d]">
        {/* Logo */}
        <div className="px-5 pt-7 pb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center bg-[rgba(19,150,186,0.1)] border border-[rgba(19,150,186,0.15)]">
              <Hand className="w-[18px] h-[18px] text-[#1396ba]" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-[#1f2328] dark:text-[#e6edf3]">
                Sign<span className="text-[#1396ba]">Interpreter</span>
              </h1>
              <p className="text-xs text-[#8b949e] dark:text-[#484f58] mt-0.5">
                V7 Research
              </p>
            </div>
          </div>
        </div>

        {/* Nav items */}
        <div className="flex-1 px-3 space-y-0.5">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `group flex items-center gap-3 px-3 h-11 rounded-lg text-base font-medium transition-colors relative ${
                  isActive
                    ? 'bg-[rgba(19,150,186,0.1)] text-[#1396ba]'
                    : 'text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22] hover:text-[#1f2328] dark:hover:text-[#e6edf3]'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-[#1396ba]" />
                  )}
                  <Icon className={`w-5 h-5 ${isActive ? 'text-[#1396ba]' : ''}`} />
                  <span>{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </div>

        {/* Theme Toggle */}
        <div className="px-3 py-3 border-t border-[#d0d7de] dark:border-[#30363d]">
          <button
            onClick={toggleTheme}
            className="w-full flex items-center gap-3 px-3 h-11 rounded-lg text-base font-medium transition-colors cursor-pointer text-[#656d76] dark:text-[#8b949e] hover:bg-[#f6f8fa] dark:hover:bg-[#161b22] hover:text-[#1f2328] dark:hover:text-[#e6edf3]"
          >
            <Sun className="w-5 h-5 hidden dark:block" />
            <Moon className="w-5 h-5 block dark:hidden" />
            <span className="block dark:hidden">Dark mode</span>
            <span className="hidden dark:block">Light mode</span>
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto pb-14 md:pb-0 bg-white dark:bg-[#0d1117]">
        <Outlet />
      </main>

      {/* Mobile Bottom Nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-white dark:bg-[#010409] border-t border-[#d0d7de] dark:border-[#30363d]">
        <div className="flex items-center justify-around px-1 py-1">
          {navItems.map(({ to, icon: Icon, label }) => {
            const isActive =
              to === '/'
                ? location.pathname === '/'
                : location.pathname.startsWith(to);
            return (
              <NavLink
                key={to}
                to={to}
                className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg transition-colors ${
                  isActive
                    ? 'text-[#1396ba]'
                    : 'text-[#8b949e] dark:text-[#484f58]'
                }`}
              >
                <Icon className="w-[18px] h-[18px]" />
                <span className="text-[10px] font-medium">{label}</span>
              </NavLink>
            );
          })}

          {/* Mobile theme toggle */}
          <button
            onClick={toggleTheme}
            className="flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer text-[#8b949e] dark:text-[#484f58]"
          >
            <Sun className="w-[18px] h-[18px] hidden dark:block" />
            <Moon className="w-[18px] h-[18px] block dark:hidden" />
            <span className="text-[10px] font-medium">Theme</span>
          </button>
        </div>
      </nav>

      <ToastContainer />
    </div>
  );
}
