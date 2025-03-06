import React from 'react';

const Card = ({ title, content, footer, children }) => {
    return (
        <div className="card">
            {title && <h3 className="card-title">{title}</h3>}
            <div className="card-body">
                {content && <p className="card-text">{content}</p>}
                {children}
            </div>
            {footer && <div className="card-footer">{footer}</div>}
        </div>
    );
};

export default Card;