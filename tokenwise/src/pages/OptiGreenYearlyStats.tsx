import React from 'react';

// Interfaces
interface StatCardProps {
  icon: string;
  value: string | number;
  label: string;
}

interface ImpactItemProps {
  icon: string;
  value: string;
  description: string;
}

interface ChartData {
  monthlyUsage: number[];
  optimizationTrends: number[];
}

interface PeriodButton {
  label: string;
  active: boolean;
}

interface HighlightData {
  title: string;
  value: string;
  subtitle: string;
}

// Static Test Data
const chartData: ChartData = {
  monthlyUsage: [45, 62, 38, 75, 55, 82, 48, 91, 67, 73, 58, 85],
  optimizationTrends: [35, 42, 48, 55, 58, 63, 67, 72, 75, 78, 82, 85]
};

const periodButtons: PeriodButton[] = [
  { label: 'Month', active: false },
  { label: 'Quarter', active: false },
  { label: 'Year', active: true }
];

const highlightData: HighlightData = {
  title: 'Total Tokens Saved in 2025',
  value: '2.4M',
  subtitle: '68% optimization rate'
};

const statCards: StatCardProps[] = [
  { icon: '💰', value: '$4,832', label: 'Cost Saved' },
  { icon: '⚡', value: '365', label: 'Days Active' },
  { icon: '🎯', value: '1,247', label: 'Requests Optimized' },
  { icon: '⏱️', value: '428h', label: 'Time Saved' }
];

const impactItems: ImpactItemProps[] = [
  { icon: '☁️', value: '892 kg', description: 'CO₂ emissions avoided' },
  { icon: '🌳', value: '14 Trees', description: 'equivalent carbon offset' },
  { icon: '⚡', value: '3,247 kWh', description: 'energy conserved' }
];

// Sub-components
const StatCard: React.FC<StatCardProps> = ({ icon, value, label }) => {
  return (
  <div className="bg-gray-50 rounded-2xl p-5 text-center">
    <div className="w-12 h-12 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center text-2xl">
      {icon}
    </div>
    <div className="text-3xl font-bold text-green-500 mb-1">{value}</div>
    <div className="text-xs text-gray-500">{label}</div>
  </div>
  );
};

const ImpactItem: React.FC<ImpactItemProps> = ({ icon, value, description }) => {
  return (
  <div className="flex items-center py-3 border-b border-green-200 last:border-b-0">
    <div className="w-11 h-11 bg-white rounded-full flex items-center justify-center text-xl mr-4">
      {icon}
    </div>
    <div className="flex-1">
      <div className="text-2xl font-bold text-green-600">{value}</div>
      <div className="text-xs text-gray-800 mt-1">{description}</div>
    </div>
  </div>
  );
};

const ChartBar: React.FC<{ height: number }> = ({ height }) => {
  return (
  <div
  className="flex-1 bg-gradient-to-t from-green-500 to-green-300 rounded-t hover:from-green-600 hover:to-green-400 transition-all cursor-pointer"
  style={{ height: `${height}%`, minHeight: '20px' }}
  />
  );
};

// Main Component
const OptiGreenYearlyStats: React.FC = () => {
  return (
  <div className="min-h-screen p-5">
    <div className="max-w-full mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-950 via-emerald-950 to-blue-950 p-8 text-center text-white">

        <p className="text-2xl opacity-90">AI Token Optimization - 2025 Overview</p>
      </div>

      {/* Period Selector */}
      <div className="flex justify-center gap-2 py-5 px-5">
        {periodButtons.map((period) => (
        <button
        key={period.label}
        className={`px-4 py-2 rounded-full text-sm font-semibold ${
        period.active
        ? 'bg-green-500 text-white'
        : 'bg-gray-200 text-gray-600'
        }`}
        >
          {period.label}
        </button>
        ))}
      </div>

      <div className="p-5">
        {/* Total Savings Highlight */}
        <div className="bg-gradient-to-r from-yellow-100 to-yellow-200 rounded-2xl p-4 mb-5 text-center">
          <div className="text-xs text-yellow-800 mb-1">{highlightData.title}</div>
          <div className="text-4xl font-bold text-yellow-600">{highlightData.value}</div>
          <div className="text-xs text-yellow-800 mt-1">{highlightData.subtitle}</div>
        </div>

        {/* Daily Token Usage Chart */}
        <div className="bg-gray-50 rounded-2xl p-5 mb-5">
          <div className="flex justify-between items-center mb-4">
            <span className="text-base font-semibold text-gray-800">Daily Token Usage</span>
            <span className="text-2xl font-bold text-green-500">8,234</span>
          </div>
          <div className="bg-white rounded-xl h-40 flex items-end justify-around p-3 gap-1 mb-3">
            {chartData.monthlyUsage.map((height, index) => (
            <ChartBar key={index} height={height} />
            ))}
          </div>
          <div className="text-md text-gray-800 text-center">Last 12 months average</div>
        </div>

        {/* Savings Cards Grid */}
        <div className="grid grid-cols-2 gap-4 mb-5 text-black">
          {statCards.map((card, index) => (
          <StatCard key={index} {...card} />
          ))}
        </div>

        {/* Environmental Impact Section */}
        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-2xl p-5 mb-5">
          <div className="text-xl font-semibold text-gray-800 mb-4 text-center">
            Environmental Impact
          </div>
          {impactItems.map((item, index) => (
          <ImpactItem key={index} {...item} />
          ))}
        </div>

        {/* Optimization Trends Chart */}
        <div className="bg-gray-50 rounded-2xl p-5 mb-5">
          <div className="flex justify-between items-center mb-4">
            <span className="text-base font-semibold text-gray-800">Optimization Trends</span>
            <span className="text-md text-green-600 font-semibold">↑ 12%</span>
          </div>
          <div className="bg-white rounded-xl h-40 flex items-end justify-around p-3 gap-1 mb-3">
            {chartData.optimizationTrends.map((height, index) => (
            <ChartBar key={index} height={height} />
            ))}
          </div>
          <div className="text-md text-gray-800 text-center">Monthly improvement rate</div>
        </div>
      </div>
    </div>
  </div>
  );
};

export default OptiGreenYearlyStats;