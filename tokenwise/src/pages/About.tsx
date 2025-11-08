const About: React.FC = () => {
    return (
        <div className="bg-gray-800 p-8 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-cyan-300 mb-4">About TokenWise</h2>
            <p className="text-gray-300 leading-relaxed">
                TokenWise helps developers analyze, optimize, and visualize token usage for LLMs.
                Learn how token quantity dominates energy and cost, while prompt complexity has
                minimal effect.
            </p>
        </div>
    );
};

export default About;
