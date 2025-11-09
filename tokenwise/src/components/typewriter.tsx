import  { useState, useEffect } from "react";

const TypewriterText = ({ text = "", speed = 120,pause=1000 ,style=""}) => {
    const [displayedText, setDisplayedText] = useState("");
    const [isDeleting, setIsDeleting] = useState(true);

    useEffect(() => {

        if (!text) return;

        let timeout;
        const handleTyping = () =>{
            if(!isDeleting){
                if(displayedText.length <  text.length){
                    setDisplayedText(text.substring(0,displayedText.length+1))
                    timeout = setTimeout(handleTyping, speed);
                }else{
                    timeout = setTimeout(() => setIsDeleting(true),pause);
                }
            }else{
                if(displayedText.length > 0){
                    setDisplayedText("")
                    timeout = setTimeout(handleTyping, speed/2);
                }else {
                    setIsDeleting(false);
                    timeout = setTimeout(handleTyping, speed);
                }
            }
        }

        timeout = setTimeout(handleTyping, speed);
        return () => clearTimeout(timeout);
    }, [displayedText,isDeleting,text, speed,pause]);

    return (
        <h1 className={style}>
            {displayedText}
            <span className="animate-pulse">|</span>
        </h1>
    );
};

export default TypewriterText;
