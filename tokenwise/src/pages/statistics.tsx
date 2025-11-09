import React from 'react';

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

export default function OptiGreenYearlyStats(): JSX.Element {
  const monthlyUsage: number[] = [45, 62, 38, 75, 55, 82, 48, 91, 67, 73, 58, 85];
  const optimizationTrends: number[] = [35, 42, 48, 55, 58, 63, 67, 72, 75, 78, 82, 85];

  return (
  <div className="min-h-screen bg-gradient-to-br from-purple-600 to-purple-900 p-5">
    <div className="max-w-md mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-green-600 p-8 text-center text-white">
        <h1 className="text-3xl font-bold mb-1">🌿 OptiGreen</h1>
        <p className="text-sm opacity-90">AI Token Optimization - 2025 Overview</p>
      </div>

      {/* Period Selector */}
      <div className="flex justify-center gap-2 py-5 px-5">
        <button className="px-4 py-2 rounded-full bg-gray-200 text-gray-600 text-sm font-semibold">
          Month
        </button>
        <button className="px-4 py-2 rounded-full bg-gray-200 text-gray-600 text-sm font-semibold">
          Quarter
        </button>
        <button className="px-4 py-2 rounded-full bg-green-500 text-white text-sm font-semibold">
          Year
        </button>
      </div>

      <div className="p-5">
        {/* Total Savings Highlight */}
        <div className="bg-gradient-to-r from-yellow-100 to-yellow-200 rounded-2xl p-4 mb-5 text-center">
          <div className="text-xs text-yellow-800 mb-1">Total Tokens Saved in 2025</div>
          <div className="text-4xl font-bold text-yellow-600">2.4M</div>
          <div className="text-xs text-yellow-800 mt-1">68% optimization rate</div>
        </div>

        {/* Daily Token Usage Chart */}
        <div className="bg-gray-50 rounded-2xl p-5 mb-5">
          <div className="flex justify-between items-center mb-4">
            <span className="text-base font-semibold text-gray-800">Daily Token Usage</span>
            <span className="text-2xl font-bold text-green-500">8,234</span>
          </div>
          <div className="bg-white rounded-xl h-40 flex items-end justify-around p-3 gap-1 mb-3">
            {monthlyUsage.map((height, index) => (
            <div
            key={index}
            className="flex-1 bg-gradient-to-t from-green-500 to-green-300 rounded-t hover:from-green-600 hover:to-green-400 transition-all cursor-pointer"
            style={{ height: `${height}%`, minHeight: '20px' }}
            />
            ))}
          </div>
          <div className="text-xs text-gray-400 text-center">Last 12 months average</div>
        </div>

        {/* Savings Cards Grid */}
        <div className="grid grid-cols-2 gap-4 mb-5">
          <div className="bg-gray-50 rounded-2xl p-5 text-center">
            <div className="w-12 h-12 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center text-2xl">
              💰
            </div>
            <div className="text-3xl font-bold text-green-500 mb-1">$4,832</div>
            <div className="text-xs text-gray-500">Cost Saved</div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-5 text-center">
            <div className="w-12 h-12 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center text-2xl">
              ⚡
            </div>
            <div className="text-3xl font-bold text-green-500 mb-1">365</div>
            <div className="text-xs text-gray-500">Days Active</div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-5 text-center">
            <div className="w-12 h-12 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center text-2xl">
              🎯
            </div>
            <div className="text-3xl font-bold text-green-500 mb-1">1,247</div>
            <div className="text-xs text-gray-500">Requests Optimized</div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-5 text-center">
            <div className="w-12 h-12 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center text-2xl">
              ⏱️
            </div>
            <div className="text-3xl font-bold text-green-500 mb-1">428h</div>
            <div className="text-xs text-gray-500">Time Saved</div>
          </div>
        </div>

        {/* Environmental Impact Section */}
        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-2xl p-5 mb-5">
          <div className="text-lg font-semibold text-gray-800 mb-4 text-center">
            Environmental Impact
          </div>

          <div className="flex items-center py-3 border-b border-green-200">
            <div className="w-11 h-11 bg-white rounded-full flex items-center justify-center text-xl mr-4">
              ☁️
            </div>
            <div className="flex-1">
              <div className="text-2xl font-bold text-green-600">892 kg</div>
              <div className="text-xs text-gray-600 mt-1">CO₂ emissions avoided</div>
            </div>
          </div>

          <div className="flex items-center py-3 border-b border-green-200">
            <div className="w-11 h-11 bg-white rounded-full flex items-center justify-center text-xl mr-4">
              🌳
            </div>
            <div className="flex-1">
              <div className="text-2xl font-bold text-green-600">14 Trees</div>
              <div className="text-xs text-gray-600 mt-1">equivalent carbon offset</div>
            </div>
          </div>

          <div className="flex items-center py-3">
            <div className="w-11 h-11 bg-white rounded-full flex items-center justify-center text-xl mr-4">
              ⚡
            </div>
            <div className="flex-1">
              <div className="text-2xl font-bold text-green-600">3,247 kWh</div>
              <div className="text-xs text-gray-600 mt-1">energy conserved</div>
            </div>
          </div>
        </div>

        {/* Optimization Trends Chart */}
        <div className="bg-gray-50 rounded-2xl p-5 mb-5">
          <div className="flex justify-between items-center mb-4">
            <span className="text-base font-semibold text-gray-800">Optimization Trends</span>
            <span className="text-sm text-green-600 font-semibold">↑ 12%</span>
          </div>
          <div className="bg-white rounded-xl h-40 flex items-end justify-around p-3 gap-1 mb-3">
            {optimizationTrends.map((height, index) => (
            <div
            key={index}
            className="flex-1 bg-gradient-to-t from-green-500 to-green-300 rounded-t hover:from-green-600 hover:to-green-400 transition-all cursor-pointer"
            style={{ height: `${height}%`, minHeight: '20px' }}
            />
            ))}
          </div>
          <div className="text-xs text-gray-400 text-center">Monthly improvement rate</div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center py-5 bg-gray-50 text-xs text-gray-400">
        OptiGreen v1.0 • Last updated: Nov 9, 2025
      </div>
    </div>
  </div>
  );
}