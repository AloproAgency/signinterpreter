import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppProvider } from './lib/context';
import Layout from './components/Layout';
import InferencePage from './pages/InferencePage';
import TeamPage from './pages/TeamPage';
import ContributionPage from './pages/ContributionPage';
import VocabularyPage from './pages/VocabularyPage';
import AdminPage from './pages/AdminPage';

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<InferencePage />} />
            <Route path="/team" element={<TeamPage />} />
            <Route path="/contribute" element={<ContributionPage />} />
            <Route path="/vocabulary" element={<VocabularyPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}
