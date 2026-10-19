import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {ArrowLeft, ArrowUpRight, BookOpen, ChevronRight, Code2, Database, LineChart, Sparkles, X} from 'lucide-react';
import './styles.css';

type Algorithm={id:string;name:string;notebook:string|null;available:boolean};
type Output={mime:string;data:string};
type Cell={index:number;cell_type:string;source:string;outputs:{output_type:string;data:Output[];ename?:string;evalue?:string}[]};
type Notebook={filename:string;category:string;cells:Record<string,Cell>;cellCount:number};

const deckConfig={
  regression:{title:'Regression',eyebrow:'FIELD MANUAL 01',sub:'Predict continuous outcomes',tone:'ochre',symbol:'∿'},
  classification:{title:'Classification',eyebrow:'FIELD MANUAL 02',sub:'Separate signal from noise',tone:'rust',symbol:'◇'},
  clustering:{title:'Clustering',eyebrow:'FIELD MANUAL 03',sub:'Discover hidden structure',tone:'sage',symbol:'✳'}
} as const;

type Category=keyof typeof deckConfig;

const fallback:Record<Category,Algorithm[]>={
  regression:['1_LinearRegression','2_RidgeRegression','3_LassoRegression','4_ElasticNetRegression','5_PolynomialRegression','6_DecisionTree','7_RandomForest','8_GradientBoosting','9_SupportVector','10_KNN'].map((id,i)=>({id,name:['Linear Regression','Ridge Regression','Lasso Regression','Elastic Net Regression','Polynomial Regression','Decision Tree','Random Forest','Gradient Boosting','Support Vector','K-Nearest Neighbors'][i],notebook:null,available:false})),
  classification:['AdaBoost','Bagging','DecisionTree','Gradient_Boosting','Knn','LogisticRegression','MLP','Naive_Bayes','RFC','SVC'].map((id,i)=>({id,name:['AdaBoost','Bagging Classifier','Decision Tree','Gradient Boosting','K-Nearest Neighbors','Logistic Regression','Multi-Layer Perceptron','Naive Bayes','Random Forest Classifier','Support Vector Classifier'][i],notebook:null,available:false})),
  clustering:[{id:'1_KmeansClustering',name:'K-Means Clustering',notebook:null,available:false},{id:'AgglomerativeClustering',name:'Agglomerative Clustering',notebook:null,available:false}]
};

// --- SVG Patent Drawing Components ---
const inkProps = {
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  fill: 'none',
};

// Larger Minimal Deck Emblems for Homepage Cards
const MinimalDeckEmblem = ({ category }: { category: Category }) => {
  if (category === 'regression') {
    return (
      <svg viewBox="0 0 100 100" style={{ width: '96px', height: '96px', color: '#2D2A26' }}>
        <path d="M 12 68 Q 35 12, 50 50 T 88 32" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" />
        <circle cx="20" cy="52" r="4" fill="currentColor" />
        <circle cx="48" cy="52" r="4" fill="currentColor" />
        <circle cx="80" cy="28" r="4" fill="currentColor" />
      </svg>
    );
  }
  if (category === 'classification') {
    return (
      <svg viewBox="0 0 100 100" style={{ width: '96px', height: '96px', color: '#2D2A26' }}>
        <line x1="18" y1="82" x2="82" y2="18" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" />
        <circle cx="30" cy="38" r="4.5" fill="currentColor" />
        <circle cx="45" cy="25" r="4.5" fill="currentColor" />
        <circle cx="58" cy="72" r="4.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
        <circle cx="72" cy="55" r="4.5" fill="none" stroke="currentColor" strokeWidth={2.2} />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 100 100" style={{ width: '96px', height: '96px', color: '#2D2A26' }}>
      <circle cx="35" cy="36" r="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3.5 3.5" />
      <circle cx="35" cy="36" r="3.5" fill="currentColor" />
      <circle cx="65" cy="36" r="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3.5 3.5" />
      <circle cx="65" cy="36" r="3.5" fill="currentColor" />
      <circle cx="50" cy="70" r="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3.5 3.5" />
      <circle cx="50" cy="70" r="3.5" fill="currentColor" />
    </svg>
  );
};

const AlgorithmDiagram = ({ id, category }: { id: string; category: Category }) => {
  const normId = id.toLowerCase();

  // REGRESSION DIAGRAMS
  if (category === 'regression' || normId.includes('regressor')) {
    if (normId.includes('linear')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 20" {...inkProps} strokeWidth={1.8} />
        <path d="M 21 26 L 25 20 L 29 26 M 169 136 L 175 140 L 169 144" fill="none" stroke="currentColor" strokeWidth={1.5} />
        <line x1="35" y1="125" x2="165" y2="35" {...inkProps} strokeWidth={2} />
        <line x1="55" y1="112" x2="55" y2="92" {...inkProps} strokeDasharray="2 2" opacity={0.6} />
        <line x1="95" y1="72" x2="95" y2="90" {...inkProps} strokeDasharray="2 2" opacity={0.6} />
        <line x1="135" y1="55" x2="135" y2="38" {...inkProps} strokeDasharray="2 2" opacity={0.6} />
        {[[42, 120], [55, 92], [75, 105], [95, 90], [115, 62], [135, 38], [152, 50]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.8" fill="currentColor" />
        ))}
        <text x="65" y="38" fontSize="10" fontFamily="serif" fontStyle="italic" fill="currentColor">y = β₀ + β₁x</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 1</text>
      </svg>
    );

    if (normId.includes('ridge')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <circle cx="100" cy="80" r="38" {...inkProps} strokeWidth={2} />
        <ellipse cx="142" cy="38" rx="42" ry="22" transform="rotate(-30 142 38)" {...inkProps} strokeDasharray="2 2" opacity={0.5} />
        <ellipse cx="142" cy="38" rx="26" ry="14" transform="rotate(-30 142 38)" {...inkProps} strokeDasharray="2 2" opacity={0.7} />
        <circle cx="142" cy="38" r="2.5" fill="currentColor" />
        <circle cx="123" cy="52" r="3.5" fill="currentColor" />
        <text x="62" y="75" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor">β₁² + β₂² ≤ t</text>
        <text x="138" y="30" fontSize="8" fontFamily="serif" fontStyle="italic" fill="currentColor">β̂_OLS</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 2</text>
      </svg>
    );

    if (normId.includes('lasso')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <polygon points="100,35 145,80 100,125 55,80" {...inkProps} strokeWidth={2} />
        <ellipse cx="145" cy="35" rx="38" ry="18" transform="rotate(-30 145 35)" {...inkProps} strokeDasharray="2 2" opacity={0.5} />
        <ellipse cx="145" cy="35" rx="22" ry="10" transform="rotate(-30 145 35)" {...inkProps} strokeDasharray="2 2" opacity={0.7} />
        <circle cx="100" cy="35" r="3.5" fill="currentColor" />
        <text x="50" y="75" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor">|β₁| + |β₂| ≤ t</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 3</text>
      </svg>
    );

    if (normId.includes('elasticnet') || normId.includes('elastic_net')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 100 37 Q 138 37 138 80 Q 138 123 100 123 Q 62 123 62 80 Q 62 37 100 37 Z" {...inkProps} strokeWidth={2} />
        <ellipse cx="142" cy="37" rx="35" ry="16" transform="rotate(-28 142 37)" {...inkProps} strokeDasharray="2 2" opacity={0.5} />
        <circle cx="118" cy="53" r="3.5" fill="currentColor" />
        <text x="52" y="75" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor">L₁ + L₂</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 4</text>
      </svg>
    );

    if (normId.includes('polynomial')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 25" {...inkProps} strokeWidth={1.8} />
        <path d="M 30 130 C 50 10, 95 160, 140 85 C 155 60, 160 30, 170 20" {...inkProps} strokeWidth={2} />
        {[[32, 122], [45, 60], [60, 42], [80, 88], [100, 128], [120, 110], [140, 80], [155, 45], [168, 25]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.8" fill="currentColor" />
        ))}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 5</text>
      </svg>
    );

    if (normId.includes('decisiontree') || normId.includes('decision_tree')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 100 25 L 60 65 M 100 25 L 140 65 M 60 65 L 40 105 M 60 65 L 80 105 M 140 65 L 120 105 M 140 65 L 160 105" {...inkProps} strokeWidth={1.8} />
        <circle cx="100" cy="25" r="9" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <circle cx="60" cy="65" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <circle cx="140" cy="65" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        {[40, 80, 120, 160].map((cx, i) => <circle key={i} cx={cx} cy="105" r="5.5" fill="currentColor" />)}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 6</text>
      </svg>
    );

    if (normId.includes('randomforest') || normId.includes('random_forest')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        {[25, 80, 135].map((tx, idx) => (
          <g key={idx} transform={`translate(${tx}, 25)`}>
            <path d="M 20 0 L 10 20 M 20 0 L 30 20 M 10 20 L 5 40 M 10 20 L 15 40 M 30 20 L 25 40 M 30 20 L 35 40" {...inkProps} strokeWidth={1.2} />
            <circle cx="20" cy="0" r="3" fill="#E8DEC9" {...inkProps} />
            {[5, 15, 25, 35].map((x, i) => <circle key={i} cx={x} cy="40" r="2" fill="currentColor" />)}
          </g>
        ))}
        <path d="M 45 70 Q 100 100 100 110 M 100 70 L 100 110 M 155 70 Q 100 100 100 110" {...inkProps} strokeDasharray="3 3" />
        <circle cx="100" cy="115" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <text x="96" y="118" fontSize="9" fontFamily="sans-serif" fontWeight="bold">∑</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 7</text>
      </svg>
    );

    if (normId.includes('gradient') || normId.includes('boosting')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 25" {...inkProps} strokeWidth={1.5} />
        <path d="M 35 120 L 70 120 L 70 80 L 110 80 L 110 50 L 165 50" {...inkProps} strokeDasharray="3 2" opacity={0.5} />
        <path d="M 35 125 C 65 110, 85 90, 110 70 C 135 50, 150 40, 165 35" {...inkProps} strokeWidth={2.2} />
        <line x1="70" y1="120" x2="70" y2="98" {...inkProps} strokeDasharray="2 2" />
        <line x1="110" y1="80" x2="110" y2="70" {...inkProps} strokeDasharray="2 2" />
        {[[45, 122], [70, 100], [90, 82], [110, 70], [135, 50], [155, 38]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.5" fill="currentColor" />
        ))}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 8</text>
      </svg>
    );

    if (normId.includes('supportvector') || normId.includes('svr') || normId.includes('support_vector')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 25" {...inkProps} strokeWidth={1.5} />
        <path d="M 35 120 Q 90 120, 120 70 T 165 35" {...inkProps} strokeWidth={2.2} />
        <path d="M 35 102 Q 90 102, 120 52 T 165 17" {...inkProps} strokeDasharray="3 3" />
        <path d="M 35 138 Q 90 138, 120 88 T 165 53" {...inkProps} strokeDasharray="3 3" />
        <circle cx="80" cy="118" r="4.5" fill="none" stroke="currentColor" strokeWidth={1.8} />
        <circle cx="120" cy="52" r="4.5" fill="none" stroke="currentColor" strokeWidth={1.8} />
        <circle cx="145" cy="62" r="4.5" fill="none" stroke="currentColor" strokeWidth={1.8} />
        {[[50, 120], [80, 118], [100, 95], [120, 52], [145, 62], [160, 32]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.5" fill="currentColor" />
        ))}
        <line x1="70" y1="102" x2="70" y2="138" {...inkProps} strokeWidth={1} />
        <text x="56" y="122" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor">2ε</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 9</text>
      </svg>
    );

    if (normId.includes('knn') || normId.includes('nearest') || normId.includes('k-nearest')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 25" {...inkProps} strokeWidth={1.5} />
        <path d="M 30 120 H 60 V 90 H 100 V 60 H 140 V 35 H 170" {...inkProps} strokeWidth={2} />
        {[[35, 128], [50, 112], [68, 98], [85, 82], [112, 68], [130, 52], [150, 42], [165, 28]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.8" fill="currentColor" />
        ))}
        <text x="120" y="125" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor">k-local mean</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 10</text>
      </svg>
    );
  }

  // CLASSIFICATION DIAGRAMS
  if (category === 'classification' || normId.includes('classifier')) {
    if (normId.includes('adaboost')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <rect x="25" y="30" width="150" height="100" rx="4" {...inkProps} strokeDasharray="3 3" opacity={0.4} />
        <line x1="85" y1="30" x2="85" y2="130" {...inkProps} strokeWidth={1.8} />
        <line x1="85" y1="80" x2="175" y2="80" {...inkProps} strokeWidth={1.8} />
        <circle cx="50" cy="55" r="6" fill="none" stroke="currentColor" strokeWidth={2} />
        <circle cx="65" cy="100" r="3" fill="none" stroke="currentColor" strokeWidth={1.5} />
        <circle cx="120" cy="50" r="7" fill="currentColor" />
        <circle cx="150" cy="60" r="3" fill="currentColor" />
        <circle cx="110" cy="105" r="3" fill="none" stroke="currentColor" strokeWidth={1.5} />
        <circle cx="140" cy="110" r="8" fill="none" stroke="currentColor" strokeWidth={2} />
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 1</text>
      </svg>
    );

    if (normId.includes('bagging')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <g transform="translate(15, 20)">
          <rect x="0" y="0" width="35" height="25" rx="2" {...inkProps} />
          <rect x="0" y="38" width="35" height="25" rx="2" {...inkProps} />
          <rect x="0" y="76" width="35" height="25" rx="2" {...inkProps} />
          <path d="M 35 12 L 75 25 M 35 50 L 75 50 M 35 88 L 75 75" {...inkProps} />
          <circle cx="85" cy="25" r="10" fill="#E8DEC9" {...inkProps} />
          <circle cx="85" cy="50" r="10" fill="#E8DEC9" {...inkProps} />
          <circle cx="85" cy="75" r="10" fill="#E8DEC9" {...inkProps} />
          <text x="81" y="28" fontSize="8" fontFamily="sans-serif">M₁</text>
          <text x="81" y="53" fontSize="8" fontFamily="sans-serif">M₂</text>
          <text x="81" y="78" fontSize="8" fontFamily="sans-serif">M₃</text>
          <path d="M 95 25 L 135 50 M 95 50 L 135 50 M 95 75 L 135 50" {...inkProps} />
          <rect x="135" y="38" width="35" height="25" rx="2" fill="currentColor" />
          <text x="140" y="53" fontSize="8" fontFamily="sans-serif" fill="#E8DEC9">Vote</text>
        </g>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 2</text>
      </svg>
    );

    if (normId.includes('decisiontree') || normId.includes('decision_tree')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 100 25 L 60 65 M 100 25 L 140 65 M 60 65 L 40 105 M 60 65 L 80 105 M 140 65 L 120 105 M 140 65 L 160 105" {...inkProps} strokeWidth={1.8} />
        <circle cx="100" cy="25" r="9" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <circle cx="60" cy="65" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <circle cx="140" cy="65" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        {[40, 80, 120, 160].map((cx, i) => <circle key={i} cx={cx} cy="105" r="5.5" fill={i % 2 === 0 ? "currentColor" : "#E8DEC9"} stroke="currentColor" strokeWidth={1.5} />)}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 3</text>
      </svg>
    );

    if (normId.includes('gradient') || normId.includes('boosting')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <rect x="25" y="25" width="150" height="110" rx="4" {...inkProps} strokeWidth={1.5} />
        <line x1="25" y1="80" x2="105" y2="80" {...inkProps} strokeDasharray="3 2" opacity={0.5} />
        <line x1="105" y1="25" x2="105" y2="135" {...inkProps} strokeDasharray="3 2" opacity={0.5} />
        <path d="M 25 90 C 70 90, 105 110, 105 80 C 105 50, 140 35, 175 35" {...inkProps} strokeWidth={2.2} />
        {[[45, 50], [70, 42], [65, 68], [40, 75]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="3" fill="currentColor" />
        ))}
        {[[125, 95], [148, 110], [160, 78], [130, 65]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="3" fill="none" stroke="currentColor" strokeWidth={1.5} />
        ))}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 4</text>
      </svg>
    );

    if (normId.includes('knn') || normId.includes('nearest') || normId.includes('k-nearest')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <circle cx="100" cy="80" r="4.5" fill="currentColor" />
        <circle cx="100" cy="80" r="45" {...inkProps} strokeDasharray="3 3" strokeWidth={1.5} />
        <line x1="100" y1="80" x2="70" y2="58" {...inkProps} />
        <line x1="100" y1="80" x2="132" y2="65" {...inkProps} />
        <line x1="100" y1="80" x2="90" y2="120" {...inkProps} />
        {[[70, 58], [132, 65], [90, 120]].map(([cx, cy], i) => <circle key={i} cx={cx} cy={cy} r="3.5" fill="currentColor" />)}
        {[[42, 32], [152, 38], [158, 112], [48, 128], [138, 132]].map(([cx, cy], i) => <circle key={i} cx={cx} cy={cy} r="3.5" fill="#E8DEC9" stroke="currentColor" strokeWidth={1.5} />)}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 5</text>
      </svg>
    );

    if (normId.includes('logistic')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 30 130 C 80 130, 120 30, 170 30" {...inkProps} strokeWidth={2.2} />
        <line x1="25" y1="30" x2="175" y2="30" {...inkProps} strokeDasharray="2 2" opacity={0.4} />
        <text x="180" y="33" fontSize="8" fontFamily="sans-serif">1.0</text>
        <text x="180" y="133" fontSize="8" fontFamily="sans-serif">0.0</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 6</text>
      </svg>
    );

    if (normId.includes('mlp') || normId.includes('perceptron') || normId.includes('multi')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        {[35, 75, 115].map(y1 => [30, 60, 90, 120].map((y2, j) => (
          <line key={`l1-${y1}-${j}`} x1="35" y1={y1} x2="95" y2={y2} {...inkProps} opacity={0.35} />
        )))}
        {[30, 60, 90, 120].map(y1 => [55, 95].map((y2, j) => (
          <line key={`l2-${y1}-${j}`} x1="95" y1={y1} x2="155" y2={y2} {...inkProps} opacity={0.35} />
        )))}
        {[35, 75, 115].map((y, i) => <circle key={i} cx="35" cy={y} r="6" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />)}
        {[30, 60, 90, 120].map((y, i) => <circle key={i} cx="95" cy={y} r="6" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />)}
        {[55, 95].map((y, i) => <circle key={i} cx="155" cy={y} r="6" fill="currentColor" />)}
        <text x="25" y="148" fontSize="8" fontFamily="monospace">Input</text>
        <text x="80" y="148" fontSize="8" fontFamily="monospace">Hidden</text>
        <text x="140" y="148" fontSize="8" fontFamily="monospace">Output</text>
      </svg>
    );

    if (normId.includes('bayes') || normId.includes('naive')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 30 135 Q 75 25, 120 135" {...inkProps} strokeWidth={2} />
        <path d="M 80 135 Q 125 45, 170 135" {...inkProps} strokeWidth={2} strokeDasharray="4 2" />
        <path d="M 80 135 Q 100 82, 120 135 Z" fill="currentColor" opacity={0.2} />
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 8</text>
      </svg>
    );

    if (normId.includes('randomforest') || normId.includes('rfc') || normId.includes('random_forest')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        {[25, 80, 135].map((tx, idx) => (
          <g key={idx} transform={`translate(${tx}, 25)`}>
            <path d="M 20 0 L 10 20 M 20 0 L 30 20 M 10 20 L 5 40 M 10 20 L 15 40 M 30 20 L 25 40 M 30 20 L 35 40" {...inkProps} strokeWidth={1.2} />
            <circle cx="20" cy="0" r="3" fill="#E8DEC9" {...inkProps} />
            {[5, 15, 25, 35].map((x, i) => <circle key={i} cx={x} cy="40" r="2" fill={i % 2 === 0 ? "currentColor" : "#E8DEC9"} stroke="currentColor" strokeWidth={1} />)}
          </g>
        ))}
        <path d="M 45 70 Q 100 100 100 110 M 100 70 L 100 110 M 155 70 Q 100 100 100 110" {...inkProps} strokeDasharray="3 3" />
        <circle cx="100" cy="115" r="8" fill="#E8DEC9" {...inkProps} strokeWidth={1.8} />
        <text x="96" y="118" fontSize="9" fontFamily="sans-serif" fontWeight="bold">∑</text>
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 9</text>
      </svg>
    );

    if (normId.includes('svc') || normId.includes('supportvector')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 25 140 L 175 140 M 25 140 L 25 25" {...inkProps} strokeWidth={1.5} />
        <line x1="45" y1="130" x2="155" y2="30" {...inkProps} strokeWidth={2.2} />
        <line x1="60" y1="140" x2="170" y2="40" {...inkProps} strokeDasharray="3 3" />
        <line x1="30" y1="120" x2="140" y2="20" {...inkProps} strokeDasharray="3 3" />
        <circle cx="68" cy="90" r="5" fill="none" stroke="currentColor" strokeWidth={2} />
        <circle cx="102" cy="55" r="5" fill="none" stroke="currentColor" strokeWidth={2} />
        <circle cx="118" cy="70" r="5" fill="none" stroke="currentColor" strokeWidth={2} />
        {[[50, 110], [42, 95], [68, 90], [118, 70]].map(([cx, cy], i) => <circle key={i} cx={cx} cy={cy} r="2.8" fill="currentColor" />)}
        {[[102, 55], [130, 45], [145, 60]].map(([cx, cy], i) => <circle key={i} cx={cx} cy={cy} r="2.8" fill="none" stroke="currentColor" strokeWidth={1.5} />)}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 10</text>
      </svg>
    );
  }

  // CLUSTERING DIAGRAMS
  if (category === 'clustering' || normId.includes('cluster')) {
    if (normId.includes('kmeans') || normId.includes('k-means')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <circle cx="58" cy="58" r="30" {...inkProps} strokeDasharray="3 3" />
        <circle cx="142" cy="58" r="30" {...inkProps} strokeDasharray="3 3" />
        <circle cx="100" cy="118" r="30" {...inkProps} strokeDasharray="3 3" />
        <path d="M 52 58 H 64 M 58 52 V 64 M 136 58 H 148 M 142 52 V 64 M 94 118 H 106 M 100 112 V 124" stroke="currentColor" strokeWidth={2} />
        {[[48, 48], [66, 68], [50, 68], [134, 50], [150, 66], [140, 44], [92, 112], [108, 124], [98, 128]].map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy} r="2.2" fill="currentColor" />
        ))}
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 1</text>
      </svg>
    );

    if (normId.includes('agglomerative') || normId.includes('hierarchical')) return (
      <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228' }}>
        <path d="M 35 130 V 90 H 75 V 130 M 125 130 V 90 H 165 V 130 M 55 90 V 40 H 145 V 90" {...inkProps} strokeWidth={2} />
        <line x1="15" y1="65" x2="185" y2="65" {...inkProps} strokeDasharray="3 3" opacity={0.6} />
        <text x="145" y="152" fontSize="9" fontFamily="serif" fontStyle="italic" fill="currentColor" opacity={0.8}>Fig 2</text>
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 200 160" style={{ width: '100%', height: '100%', color: '#3A3228', opacity: 0.6 }}>
      <circle cx="100" cy="80" r="4" fill="currentColor" />
    </svg>
  );
};

function App(){
 const [algorithms,setAlgorithms]=useState<Record<Category,Algorithm[]>>(fallback);
 const [category,setCategory]=useState<Category|null>(null);
 const [selected,setSelected]=useState<Algorithm|null>(null);
 const [tab,setTab]=useState<'code'|'graphs'|'info'>('code');
 const [nb,setNb]=useState<Notebook|null>(null);
 const [info,setInfo]=useState<any>(null);
 const [busy,setBusy]=useState(false);
 const [error,setError]=useState('');

 useEffect(()=>{fetch('/api/algorithms').then(r=>r.ok?r.json():Promise.reject()).then(d=>setAlgorithms(d)).catch(()=>{});},[]);
 useEffect(()=>{
  if(!selected)return;
  setNb(null);setInfo(null);setError('');setBusy(true);
  if(selected.notebook){
    fetch(`/api/notebook/${category}/${encodeURIComponent(selected.notebook)}`)
      .then(r=>r.ok?r.json():Promise.reject(new Error('Notebook could not be loaded')))
      .then(setNb)
      .catch(e=>setError(e.message))
      .finally(()=>setBusy(false));
  }else{
    setBusy(false);
    setError('No matching notebook found. Add the notebook to normalised_notebooks/'+category+'/ and refresh.');
  }
  fetch(`/api/info/${encodeURIComponent(selected.name)}`).then(r=>r.ok?r.json():null).then(setInfo).catch(()=>{});
 },[selected,category]);

 const chooseDeck=(c:Category)=>{setCategory(c);setSelected(null);setTab('code');};
 const goBack=()=>{if(selected){setSelected(null);setNb(null);setInfo(null);}else setCategory(null);};
 const openCard=(a:Algorithm)=>{setSelected(a);setTab('code');};

 return <main className={category?'table-mode':'landing-mode'}>
  {!category ? (
    <>
      <header className="topbar">
        <div className="brand">Field Manuals <span>· Machine Learning</span></div>
        <div className="top-actions">
          <button className="collection" onClick={()=>document.getElementById('collection')?.scrollIntoView({behavior:'smooth'})}>
            THE COLLECTION <ArrowUpRight size={14}/>
          </button>
        </div>
      </header>
      <section className="hero">
        <div className="hero-kicker"><span>23CSE301</span><span>·</span><span>Machine Learning Capstone</span></div>
        <h1>
  MACHINE LEARNING
  <br />
  <em>Capstone Project</em>
</h1>
        
        {/* Restored Divider Rule and Heading */}
        <div className="hero-rule"><span>SELECT A DECK TO BEGIN</span><span>01 — 03</span></div>
        
        {/* Homepage Stage Layout with Enlarged Emblem Icons */}
        <div id="collection" className="deck-stage">
          {(Object.keys(deckConfig) as Category[]).map((c,i)=>(
            <button 
              key={c} 
              className={`hero-deck ${deckConfig[c].tone} deck-${i+1}`} 
              onClick={()=>chooseDeck(c)} 
              aria-label={`Open ${deckConfig[c].title} deck`}
            >
              <span className="deck-grain"/>
              <span className="deck-topline">{deckConfig[c].eyebrow}</span>
              
              {/* Scaled-up Minimal Vector Emblem */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', margin: 'auto' }}>
                <MinimalDeckEmblem category={c} />
              </div>

              <span className="deck-name">{deckConfig[c].title}</span>
              <span className="deck-sub">{deckConfig[c].sub}</span>
              <span className="deck-index">ML / {String(i+1).padStart(2,'0')}</span>
              <span className="deck-corner">✳</span>
            </button>
          ))}
        </div>

        <div className="landing-foot"><span>BUILT FROM YOUR NOTEBOOKS</span><span>SCROLL / SELECT / EXPLORE</span></div>
      </section>
    </>
  ) : (
    <>
      <header className="tablebar">
        <button className="back-link" onClick={goBack}><ArrowLeft size={16}/> {selected?'BACK TO TABLE':'ALL DECKS'}</button>
        <div className="table-brand">
  MACHINE LEARNING CAPSTONE PROJECT <span> / </span> {deckConfig[category].title.toUpperCase()}
</div>
        <div className="table-count">{selected?'FIELD NOTES':`${algorithms[category].length} CARDS`}</div>
      </header>
      {!selected ? (
        <section className="table-content">
          <div className="table-heading">
            <div>
              <div className="eyebrow">{deckConfig[category].eyebrow} · COLLECTION</div>
              <h2>{deckConfig[category].title}<i> deck</i></h2>
              <p>{deckConfig[category].sub}. Select any card to inspect the model.</p>
            </div>
          </div>
          <div className={`dealt-grid ${category==='clustering'?'two-cards':''}`}>
            {algorithms[category].map((a,i)=>(
              <button key={a.id} className={`algorithm-card ${deckConfig[category].tone}`} style={{'--i':i} as React.CSSProperties} onClick={()=>openCard(a)}>
                <div className="card-inner">
                  <div className="card-minihead">
                    <span>ML FIELD CARD</span>
                    <span>{String(i+1).padStart(2,'0')}</span>
                  </div>
                  <div className="card-art">
                    <div className="art-axis axis-x"/>
                    <div className="art-axis axis-y"/>
                    
                    <div style={{ width: '100%', height: '100%', padding: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
                      <AlgorithmDiagram id={a.id} category={category} />
                    </div>
                  </div>
                  <div className="card-label">
                    <span className="card-kind">{category.toUpperCase()} / MODEL</span>
                    <strong>{a.name}</strong>
                    <span className="card-open">OPEN FIELD NOTES <ChevronRight size={13}/></span>
                  </div>
                  <div className="card-bottom">
                    <span>CAPSTONE ARCHIVE</span>
                    <span>✳</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
          <div className="table-bottom">
            <span>CARDS REMAIN ON THE TABLE AFTER VIEWING</span>
            <span>SELECT A CARD TO REVEAL ↗</span>
          </div>
        </section>
      ) : (
        <section className="detail-layout">
          <div className="reveal-column">
            <div className="reveal-label">
              <span>FIELD CARD / {category.toUpperCase()}</span>
              <button className="icon-button" onClick={goBack} aria-label="Close card"><X size={17}/></button>
            </div>
            <div className={`revealed-card ${deckConfig[category].tone}`}>
              <div className="revealed-frame">
                <div className="reveal-top">
                  <span>ML FIELD CARD</span>
                  <span>NO. {String(algorithms[category].findIndex(a=>a.id===selected.id)+1).padStart(2,'0')}</span>
                </div>
                <div className="reveal-art">
                  <div className="big-axis"/>
                  
                  <div style={{ width: '100%', height: '100%', padding: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
                    <AlgorithmDiagram id={selected.id} category={category} />
                  </div>
                  
                  <div className="art-caption">MODEL STUDY · {category.toUpperCase()}</div>
                </div>
                <div className="revealed-name">
                  <small>ALGORITHM</small>
                  <h2>{selected.name}</h2>
                  <p>OBJECTIVE · METHOD · EVIDENCE</p>
                </div>
                <div className="revealed-footer">
                  <span>23CSE301 / ML CAPSTONE</span>
                  <span>✳</span>
                </div>
              </div>
            </div>
            <div className="reveal-caption"><Sparkles size={14}/> A closer look at the method</div>
          </div>
          <div className="detail-panel">
            <div className="detail-heading">
              <div className="eyebrow">FIELD NOTES / {selected.name.toUpperCase()}</div>
              <h1>{selected.name}</h1>
              <p>Notebook-backed implementation, saved outputs, and model reference.</p>
            </div>
            <div className="tabs">
              <button className={tab==='code'?'active':''} onClick={()=>setTab('code')}><Code2 size={15}/> Code</button>
              <button className={tab==='graphs'?'active':''} onClick={()=>setTab('graphs')}><LineChart size={15}/> Graphs</button>
              <button className={tab==='info'?'active':''} onClick={()=>setTab('info')}><BookOpen size={15}/> Info</button>
            </div>
            <div className="panel-scroll">
              {busy ? (
                <div className="loading-state"><span className="loader"/> Reading notebook…</div>
              ) : error && tab!=='info' ? (
                <div className="empty-state"><Database size={24}/><h3>Notebook not available</h3><p>{error}</p></div>
              ) : tab==='code' ? (
                <>
                  <div className="section-intro">
                    <span>01 — SOURCE & RESULTS</span>
                    <p>Cells 10, 12 and 14 from the standardized notebook, with their saved outputs.</p>
                  </div>
                  {[10,12,14].map(i=><CellView key={i} cell={nb?.cells[String(i)]} title={i===10?'Model Training':i===12?'Model Evaluation':'Results & Conclusions'}/>)}
                </>
              ) : tab==='graphs' ? (
                <>
                  <div className="section-intro">
                    <span>02 — VISUAL EVIDENCE</span>
                    <p>Saved exploratory plots from cell 6 and evaluation outputs from cell 12.</p>
                  </div>
                  <CellView cell={nb?.cells['6']} title="Exploratory Data Analysis · Cell 6"/>
                  <CellView cell={nb?.cells['12']} title="Model Evaluation · Cell 12"/>
                </>
              ) : (
                <>
                  <div className="section-intro">
                    <span>03 — MODEL REFERENCE</span>
                    <p>Conceptual reference for understanding the method. Results and configuration remain notebook-specific.</p>
                  </div>
                  {info ? (
                    <div className="info-stack">
                      <InfoBlock label="OBJECTIVE" value={info.objective}/>
                      <InfoBlock label="OBJECTIVE FUNCTION" value={info.function}/>
                      <InfoBlock label="ASSUMPTIONS & LIMITATIONS" value={info.limitations}/>
                      <InfoBlock label="TYPICAL USE CASES" value={info.uses}/>
                      <div className="source-note">General algorithm reference · Confirm the exact criterion, loss, and hyperparameters in the notebook.</div>
                    </div>
                  ) : (
                    <div className="loading-state">Loading algorithm reference…</div>
                  )}
                </>
              )}
            </div>
            <div className="detail-foot">
              <span>{selected.notebook||'NO NOTEBOOK MATCHED'}</span>
              <span>{nb?`${nb.cellCount} CELLS`:'LOCAL NOTEBOOK API'}</span>
            </div>
          </div>
        </section>
      )}
    </>
  )}
 </main>;
}

function InfoBlock({label,value}:{label:string;value:string}){
  return <article className="info-block"><div>{label}</div><p>{value}</p></article>;
}

function CellView({cell,title}:{cell?:Cell;title:string}){
  return (
    <article className="cell-card">
      <header><span>{title}</span><span>{cell?`CELL ${cell.index}`:'NOT FOUND'}</span></header>
      {cell?.source ? (
        <pre className="code-block"><code>{cell.source}</code></pre>
      ) : (
        <div className="cell-empty">No source found for this cell.</div>
      )}
      {cell?.outputs?.length ? (
        <div className="outputs">
          <div className="output-label">SAVED OUTPUT · {cell.outputs.length}</div>
          {cell.outputs.map((o,oi)=>(
            <React.Fragment key={oi}>
              {o.data.map((d,di)=>
                d.mime.startsWith('image/') ? (
                  <img className="output-image" key={di} src={d.data} alt={`Notebook output ${di+1}`}/>
                ) : d.mime==='text/html' ? (
                  <div className="html-output" key={di} dangerouslySetInnerHTML={{__html:d.data}}/>
                ) : (
                  <pre className="text-output" key={di}>{d.data}</pre>
                )
              )}
            </React.Fragment>
          ))}
        </div>
      ) : (
        <div className="cell-empty">No saved output for this cell.</div>
      )}
    </article>
  );
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);