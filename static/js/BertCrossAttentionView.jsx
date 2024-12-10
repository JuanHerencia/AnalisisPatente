// BertCrossAttentionView.jsx
const BertCrossAttentionView = ({ mainPatent, citedPatent }) => {
    const [bertData, setBertData] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false);
    const [error, setError] = React.useState(null);
    const [selectedToken, setSelectedToken] = React.useState(null);
  
    React.useEffect(() => {
      const loadBertAnalysis = async () => {
        if (!mainPatent?.text || !citedPatent?.text) {
          return;
        }
  
        setIsLoading(true);
        try {
          const response = await fetch('/api/visualization/bert', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              main_text: mainPatent.text,
              cited_text: citedPatent.text
            })
          });
  
          if (!response.ok) {
            throw new Error('Error en el análisis BERT');
          }
  
          const data = await response.json();
          console.log("Datos recibidos del análisis BERT:", data);
          setBertData(data.analysis);
          setError(null);
        } catch (err) {
          console.error("Error en la carga del análisis BERT:", err);
          setError(err.message);
        } finally {
          setIsLoading(false);
        }
      };
  
      loadBertAnalysis();
    }, [mainPatent, citedPatent]);
  
    const getAttentionColor = (score) => {
      const intensity = Math.floor(score * 255);
      return `rgba(0, 0, ${intensity}, ${score})`;
    };
  
    const TokenContainer = ({ token, index, isSpecial, isMain, attentionScores }) => {
      const isSelected = selectedToken === index && isMain;
      const hasAttention = selectedToken !== null && attentionScores !== null;
      
      const style = {
        backgroundColor: hasAttention ? getAttentionColor(attentionScores) : 'white',
        border: isSpecial ? '2px solid rgba(0, 0, 0, 0.2)' : 'none',
        cursor: isMain ? 'pointer' : 'default',
        padding: '4px 8px',
        margin: '2px',
        borderRadius: '4px',
        display: 'inline-block',
        transition: 'all 0.3s ease',
      };
  
      if (isSelected) {
        style.backgroundColor = 'rgba(0, 0, 0, 0.1)';
      }
  
      return (
        <span
          className="inline-block transition-all duration-300 hover:bg-gray-100"
          style={style}
          onClick={() => isMain && setSelectedToken(isSelected ? null : index)}
          title={`${token}${hasAttention ? ` - Atención: ${(attentionScores * 100).toFixed(1)}%` : ''}`}
        >
          {token}
        </span>
      );
    };
  
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-blue-500">Analizando textos con BERT...</div>
        </div>
      );
    }
  
    if (error) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-red-500">{error}</div>
        </div>
      );
    }
  
    if (!bertData) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-gray-500">Seleccione una patente para comparar</div>
        </div>
      );
    }
  
    return (
      <div className="w-full h-full overflow-auto p-4 space-y-6">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          {/* Header */}
          <div className="border-b border-gray-200 bg-gray-50 p-4">
            <h2 className="text-xl font-semibold text-gray-800">
              Análisis de Atención Cruzada BERT
            </h2>
          </div>
  
          {/* Content */}
          <div className="p-4 space-y-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Patente Principal: {mainPatent.id}</h3>
              <div className="p-4 bg-gray-50 rounded text-wrap break-words">
                {bertData.text1_tokens.map((token, idx) => (
                  <TokenContainer
                    key={idx}
                    token={token}
                    index={idx}
                    isSpecial={bertData.is_special1[idx]}
                    isMain={true}
                    attentionScores={
                      selectedToken === idx ? null :
                      selectedToken !== null ? bertData.cross_attention[selectedToken][idx] : null
                    }
                  />
                ))}
              </div>
            </div>
  
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Patente Citada: {citedPatent.id}</h3>
              <div className="p-4 bg-gray-50 rounded text-wrap break-words">
                {bertData.text2_tokens.map((token, idx) => (
                  <TokenContainer
                    key={idx}
                    token={token}
                    index={idx}
                    isSpecial={bertData.is_special2[idx]}
                    isMain={false}
                    attentionScores={
                      selectedToken !== null ? bertData.cross_attention[selectedToken][idx] : null
                    }
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  if (typeof window !== 'undefined') {
    window.BertCrossAttentionView = BertCrossAttentionView;
  }