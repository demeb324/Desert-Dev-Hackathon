import React, { useState } from 'react';

export default function About() {
  const [hoveredStat, setHoveredStat] = useState(null);
  const [hoveredFeature, setHoveredFeature] = useState(null);

  const stats = [
    { number: '45%', label: 'Token Reduction' },
    { number: '38%', label: 'Energy Savings' },
    { number: '1.2M', label: 'Tons CO₂ Saved Annually' },
    { number: '98%', label: 'Quality Maintained' },
  ];

  const features = [
    {
      icon: '🤖',
      title: 'Smart Compression',
      description: 'AI-powered algorithms intelligently compress prompts without losing semantic meaning or context.',
    },
    {
      icon: '⚡',
      title: 'Real-Time Optimization',
      description: 'Instant token reduction with sub-second processing time, seamlessly integrated into your workflow.',
    },
    {
      icon: '🌍',
      title: 'Climate Impact',
      description: 'Every optimized query contributes to reducing global AI carbon footprint and energy consumption.',
    },
    {
      icon: '📊',
      title: 'Performance Analytics',
      description: 'Track your token savings, energy reduction, and environmental impact in real-time dashboards.',
    },
    {
      icon: '🔒',
      title: 'Quality Assurance',
      description: 'Maintains 98%+ response quality while achieving significant token and energy reductions.',
    },
    {
      icon: '🚀',
      title: 'Easy Integration',
      description: 'Simple API integration with popular AI platforms and frameworks in minutes.',
    },
  ];

  const impactMetrics = [
    { icon: '🏠', text: '250,000 homes powered annually' },
    { icon: '🌳', text: '50 million trees planted equivalent' },
    { icon: '🚗', text: '260,000 cars off the road' },
    { icon: '💡', text: '4.2 TWh energy saved yearly' },
    { icon: '🌡️', text: '1.2M tons CO₂ emissions prevented' },
    { icon: '💰', text: '$420M in energy costs saved' },
    { icon: '🔋', text: '38% reduction in GPU runtime' },
    { icon: '❄️', text: '30% less cooling energy needed' },
  ];

  const climateTechImpacts = [
    {
      icon: '🌊',
      title: 'Water Savings',
      text: '1.5 billion gallons saved annually (data center cooling)',
    },
    {
      icon: '🏗️',
      title: 'Infrastructure',
      text: 'Delays need for 3-4 new data centers annually',
    },
    {
      icon: '🎯',
      title: 'Carbon Goals',
      text: 'Helps companies meet 2030 net-zero commitments',
    },
  ];

  return (
  <div className="min-h-screen bg-gradient-to-br from-slate-950 via-emerald-950 to-blue-950 text-white p-5">
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <header className="py-10 text-center">
        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent mb-2">
          OptiGreen
        </h1>
        <p className="text-xl text-cyan-400 font-light ">
          Optimizing Today for a Greener Tomorrow
        </p>
      </header>

      {/* Hero Section */}
      <div className="bg-white bg-opacity-5 backdrop-blur-sm rounded-3xl p-12 my-10 border border-white border-opacity-10">
        <h2 className="text-5xl md:text-6xl font-bold text-center mb-5 leading-tight text-black">
          Reduce <span className="text-green-400">45%</span> of AI Energy Consumption
        </h2>
        <p className="text-xl md:text-2xl text-black text-center max-w-4xl mx-auto leading-relaxed">
          Our intelligent token optimization reduces computational waste, cutting energy usage and carbon emissions while maintaining AI performance quality.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 my-16 ">
        {stats.map((stat, index) => (
        <div
        key={index}
        className="bg-green-400 bg-opacity-10 border-l-4 border-green-400 p-8 rounded-lg transition-transform duration-300 hover:-translate-y-2 cursor-pointer"
        onMouseEnter={() => setHoveredStat(index)}
        onMouseLeave={() => setHoveredStat(null)}
        >
          <div className="text-5xl font-bold text-black mb-2">
            {stat.number}
          </div>
          <div className="text-lg text-black">{stat.label}</div>
        </div>
        ))}
      </div>

      {/* Visualization */}
      <div className="bg-white bg-opacity-5 rounded-2xl p-10 my-10">
        <h3 className="text-4xl text-center mb-8 text-green-400">How It Works</h3>
        <div className="flex flex-wrap justify-around items-center gap-10">
          {/* Before */}
          <div className="text-center flex-1 min-w-[200px]">
            <div className="w-44 h-44 mx-auto mb-5 rounded-full flex items-center justify-center text-5xl font-bold bg-gradient-to-br from-red-500 to-orange-600 shadow-lg shadow-red-500/50">
              1000
            </div>
            <div className="text-xl mt-2 text-cyan-800">Traditional AI Query</div>
            <div className="text-yellow-400 mt-2">⚡ High Energy</div>
          </div>

          {/* Arrow */}
          <div className="text-6xl text-green-400">→</div>

          {/* After */}
          <div className="text-center flex-1 min-w-[200px]">
            <div className="w-44 h-44 mx-auto mb-5 rounded-full flex items-center justify-center text-5xl font-bold bg-gradient-to-br from-emerald-400 to-cyan-400 shadow-lg shadow-emerald-400/50">
              550
            </div>
            <div className="text-xl mt-2 text-cyan-800">Optimized Query</div>
            <div className="text-green-400 mt-2">🌱 Low Energy</div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 my-16">
        {features.map((feature, index) => (
        <div
        key={index}
        className={`bg-white bg-opacity-5 p-8 rounded-2xl border transition-all duration-300 cursor-pointer ${
        hoveredFeature === index
        ? 'border-green-400 shadow-lg shadow-green-400/20'
        : 'border-green-400 border-opacity-20'
        }`}
        onMouseEnter={() => setHoveredFeature(index)}
        onMouseLeave={() => setHoveredFeature(null)}
        >
          <div className="text-5xl mb-5">{feature.icon}</div>
          <h3 className="text-2xl mb-4 text-green-400">{feature.title}</h3>
          <p className="text-cyan-800 leading-relaxed">{feature.description}</p>
        </div>
        ))}
      </div>

      {/* Impact Section */}
      <div className="bg-gradient-to-br from-green-400/10 to-blue-500/10 rounded-3xl p-12 my-10">
        <h2 className="text-4xl md:text-5xl mb-8 text-green-400 text-center">Environmental Impact at Scale</h2>

        <p className="text-lg md:text-xl text-white max-w-4xl mx-auto mb-5 text-center">
          AI infrastructure consumes 10-20% of data center energy globally. With AI queries projected to reach 9 trillion by 2030, the carbon impact is exponential without optimization.
        </p>

        <p className="text-base md:text-lg text-white max-w-4xl mx-auto mb-10 text-center">
          <strong>OptiGreen's Solution:</strong> By reducing token processing by 45%, we directly cut GPU computation time, energy draw, and cooling requirements—addressing the full data center energy stack.
        </p>

        {/* Scale & Significance Box */}
        <div className="bg-green-400 bg-opacity-10 rounded-2xl p-8 my-10 max-w-4xl mx-auto">
          <h3 className="text-black text-2xl md:text-3xl mb-5 text-center">Scale & Significance</h3>
          <div className="text-left text-blue-800 space-y-4 text-base md:text-lg leading-relaxed">
            <p>📈 <strong className="text-black">Market Scale:</strong> ChatGPT alone handles 10M+ daily queries. Optimizing just 1% of global AI traffic saves 42 GWh annually.</p>
            <p>🌍 <strong className="text-black">Global Impact:</strong> Training GPT-3 emitted 552 tons of CO₂. OptiGreen reduces inference emissions—the ongoing, cumulative impact affecting billions of queries.</p>
            <p>⚡ <strong className="text-black">Clean Energy Alignment:</strong> Even renewable-powered data centers benefit—reduced load means more clean energy available for other applications.</p>
            <p>🏭 <strong className="text-black">Infrastructure Relief:</strong> Lower computational demands reduce pressure on grid infrastructure and data center expansion.</p>
          </div>
        </div>

        <h3 className="text-3xl md:text-4xl my-12 text-green-400 text-center">If Deployed Across Major AI Platforms</h3>

        {/* Impact Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8 my-10">
          {impactMetrics.map((metric, index) => (
          <div key={index} className="p-3 md:p-5 text-center">
            <div className="text-4xl md:text-5xl mb-4">{metric.icon}</div>
            <div className="text-sm md:text-lg text-cyan-100">
              <strong>{metric.text.split(' ')[0]}</strong> {metric.text.split(' ').slice(1).join(' ')}
            </div>
          </div>
          ))}
        </div>

        {/* Climate Tech Impact */}
        <div className="bg-gradient-to-br from-blue-500/20 to-green-400/20 rounded-2xl p-8 md:p-10 my-12 max-w-4xl mx-auto">
          <h3 className="text-green-400 text-2xl md:text-3xl mb-8 text-center">Real-World Climate Tech Impact</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {climateTechImpacts.map((impact, index) => (
            <div key={index} className="text-center">
              <div className="text-4xl md:text-5xl mb-3">{impact.icon}</div>
              <div className="text-cyan-100 leading-relaxed text-sm md:text-base">
                <strong>{impact.title}</strong><br />
                {impact.text}
              </div>
            </div>
            ))}
          </div>
        </div>

        {/* Why This Matters Now */}
        <div className="mt-12 p-6 md:p-8 bg-white bg-opacity-5 rounded-2xl border-l-4 border-green-400 max-w-4xl mx-auto">
          <h3 className="text-green-400 text-2xl md:text-3xl mb-4">Why This Matters Now</h3>
          <p className="text-purple-500 text-base md:text-lg leading-relaxed">
            AI energy consumption is projected to <strong>double every 3.4 months</strong> as models grow larger and adoption accelerates.
            Traditional solutions focus on hardware efficiency—OptiGreen addresses the problem at the software level, creating
            immediate, scalable impact without requiring infrastructure overhaul. Every query optimized today compounds into
            measurable climate benefit tomorrow.
          </p>
        </div>
      </div>

      {/* CTA Section */}
      <div className="text-center py-16">
        <h2 className="text-4xl md:text-5xl mb-5">Join the Green AI Revolution</h2>
        <p className="text-lg md:text-xl text-cyan-100 mb-10">
          Make your AI applications more sustainable today
        </p>
        <button className="px-8 md:px-12 py-4 md:py-5 bg-gradient-to-r from-emerald-400 to-cyan-400 text-white rounded-full text-xl md:text-2xl font-semibold transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl shadow-lg shadow-emerald-400/30">
          Try OptiGreen
        </button>
      </div>
    </div>
  </div>
  );
}
