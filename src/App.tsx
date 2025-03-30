import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Navbar from './components/navbar';
import SustainabilityDashboard from './components/SustainabilityDashboard';
import FinancialDashboard from './components/FinancialDashboard';
import Footer from './components/Footer';
import LandingPage from './components/LandingPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-farmer-50">
        <Navbar />
        <main className="mx-auto">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/sustainability" element={<SustainabilityDashboard />} />
            <Route path="/financial" element={<FinancialDashboard />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;