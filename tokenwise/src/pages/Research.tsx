import React, { useState } from 'react';

export default function Research() {
  const [selectedReport, setSelectedReport] = useState<string>('comparison');

  const reports = [
    {
      id: 'comparison',
      title: 'Comparison Power Report',
      description: 'Comparative analysis of original, rule-optimized, and LLM-optimized prompts',
      filename: 'comparison_power_report.pdf',
      date: 'November 8, 2025',
      icon: '📊'
    },
    {
      id: 'benchmark',
      title: 'LM Studio Benchmark Report',
      description: 'Detailed power consumption benchmark for LM Studio inference',
      filename: 'lm_studio_benchmark_20251108_145647_power_report.pdf',
      date: 'November 8, 2025',
      icon: '⚡'
    }
  ];

  const activeReport = reports.find(r => r.id === selectedReport) || reports[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-emerald-950 to-blue-950 text-white p-5">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="py-10 text-center">
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            Research Reports
          </h1>
          <p className="text-xl text-cyan-400 font-light">
            LLM Power Consumption Analysis & Benchmarks
          </p>
        </header>

        {/* Report Selector */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          {reports.map((report) => (
            <button
              key={report.id}
              onClick={() => setSelectedReport(report.id)}
              className={`flex-1 p-6 rounded-2xl border-2 transition-all duration-300 text-left ${
                selectedReport === report.id
                  ? 'bg-emerald-500 bg-opacity-20 border-emerald-400 shadow-lg shadow-emerald-400/30'
                  : 'bg-white bg-opacity-5 border-white border-opacity-20 hover:border-cyan-400'
              }`}
            >
              <div className="text-4xl mb-3">{report.icon}</div>
              <h3 className="text-xl font-bold text-emerald-400 mb-2">{report.title}</h3>
              <p className="text-sm text-gray-300 mb-2">{report.description}</p>
              <p className="text-xs text-cyan-400">{report.date}</p>
            </button>
          ))}
        </div>

        {/* PDF Viewer */}
        <div className="bg-white bg-opacity-5 backdrop-blur-sm rounded-3xl p-8 border border-white border-opacity-10">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-emerald-400">{activeReport.title}</h2>
            <a
              href={`/reports/${activeReport.filename}`}
              download
              className="px-6 py-2 bg-gradient-to-r from-emerald-400 to-cyan-400 text-white rounded-full font-semibold hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
            >
              Download PDF
            </a>
          </div>

          <div className="w-full" style={{ height: 'calc(100vh - 400px)', minHeight: '600px' }}>
            <iframe
              src={`/reports/${activeReport.filename}`}
              className="w-full h-full border-2 border-cyan-400 rounded-lg"
              title={activeReport.title}
            />
          </div>
        </div>

        {/* Methodology Note */}
        <div className="mt-8 p-6 bg-white bg-opacity-5 rounded-2xl border-l-4 border-emerald-400">
          <h3 className="text-emerald-400 text-xl mb-3">About These Reports</h3>
          <p className="text-gray-300 leading-relaxed">
            These scientific reports document our research into LLM power consumption optimization.
            The benchmarks compare three prompt optimization strategies: original prompts with polite
            lexicon, rule-based optimization (regex removal of politeness), and LLM-based optimization.
            All measurements were collected using macOS powermetrics on Apple Silicon hardware with
            microsecond-precision timing correlation.
          </p>
        </div>
      </div>
    </div>
  );
}
