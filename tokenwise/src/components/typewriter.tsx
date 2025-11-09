import { useState, useEffect } from "react";
import React from 'react'; // Import React

const TypewriterText: React.FC<{
    text?: string;
    speed?: number;
    pause?: number;
    style?: string
}> = ({ text = "", speed = 120, pause = 1000, style = "" }) => {

    const [displayedText, setDisplayedText] = useState("");
    // 💡 Add state to track if the *typing* phase is complete (not just deleting)
    const [isComplete, setIsComplete] = useState(false);
    const [isDeleting, setIsDeleting] = useState(true);

    useEffect(() => {
        if (!text) return;

        let timeout: NodeJS.Timeout;

        const handleTyping = () => {
            if (!isDeleting) {
                // --- TYPING PHASE ---
                if (displayedText.length < text.length) {
                    // Continue typing
                    setDisplayedText(text.substring(0, displayedText.length + 1));
                    timeout = setTimeout(handleTyping, speed);
                } else {
                    // Typing is complete, start pause before deletion
                    setIsComplete(true); // 💡 Set completion flag here
                    timeout = setTimeout(() => {
                        setIsDeleting(true);
                        setIsComplete(false); // Reset completion flag when deleting starts
                    }, pause);
                }
            } else {
                // --- DELETING PHASE ---
                if (displayedText.length > 0) {
                    // Continue deleting
                    setIsComplete(true);
                    // Use a faster speed for deleting
                    timeout = setTimeout(handleTyping, speed / 2);
                } else {
                    // Deletion is complete, start typing again
                    setIsDeleting(false);
                    // Add a small pause before restarting the typing effect
                    timeout = setTimeout(handleTyping, speed);
                }
            }
        };

        // Start the effect
        // Note: The speed parameter is used for the initial delay here, which is standard.
        timeout = setTimeout(handleTyping, speed);

        return () => clearTimeout(timeout);
    }, [displayedText, isDeleting, text, speed, pause]);

    // 💡 Conditional Rendering for the Cursor
    return (
        <h1 className={style}>
            {displayedText}
            {/* Display the cursor only when the typing phase is NOT complete */}
            {!isComplete && (
                <span className="animate-pulse" id="animate">|</span>
            )}
        </h1>
    );
};

export default TypewriterText;