import { useApp } from '../lib/context';
import { X } from 'lucide-react';

const ACCENT: Record<string, string> = {
  success: 'bg-emerald-500',
  error:   'bg-red-500',
  info:    'bg-blue-500',
  warning: 'bg-amber-400',
};

export default function ToastContainer() {
  const { toasts, removeToast } = useApp();
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-100 flex flex-col gap-2 max-w-sm pointer-events-none">
      {toasts.map(toast => (
        <div
          key={toast.id}
          className="animate-toast-in pointer-events-auto flex items-center gap-3 pr-3 pl-0 py-3
            bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700
            rounded-lg shadow-lg overflow-hidden"
        >
          <div className={`w-1 self-stretch shrink-0 ${ACCENT[toast.type] ?? 'bg-zinc-400'}`} />
          <span className="flex-1 text-sm text-zinc-800 dark:text-zinc-200 pl-2">{toast.message}</span>
          <button
            onClick={() => removeToast(toast.id)}
            className="p-1 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors cursor-pointer rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
