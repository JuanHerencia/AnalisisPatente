const SemanticComparisonView = ({ mainPatent, citedPatent }) => {
  const [semanticData, setSemanticData] = React.useState(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    const loadSemanticData = async () => {
      if (!mainPatent?.text || !citedPatent?.text) {
        setIsLoading(false);
        setError('Seleccione una patente citada y asegúrese que tenga texto');
        return;
      }

      setIsLoading(true);
      try {
        const requestData = {
          main_text: mainPatent.text,
          cited_text: citedPatent.text
        };

        console.log('Enviando datos al servidor:', requestData);

        const response = await fetch('/api/visualization/semantic', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(requestData)
        });

        const responseText = await response.text();
        console.log('Respuesta del servidor:', responseText);

        if (!response.ok) {
          throw new Error(responseText);
        }

        const data = JSON.parse(responseText);
        setSemanticData(data);
        setError(null);
      } catch (err) {
        console.error('Error en análisis semántico:', err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    loadSemanticData();
  }, [mainPatent, citedPatent]);

  if (!mainPatent?.text || !citedPatent?.text) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">Seleccione una patente citada o asegúrese que los textos estén disponibles</p>
      </div>
    );
  }

  // Render del componente cuando hay error
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-red-500 text-center p-4 bg-red-50 rounded-lg">
          <h3 className="font-bold mb-2">Error en el análisis</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  // Render del componente cuando está cargando
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-blue-500">Cargando análisis semántico...</div>
      </div>
    );
  }

  // Render del componente cuando hay datos
  if (semanticData) {
    return (
      <div className="w-full h-full overflow-auto bg-white p-4 rounded-lg shadow">
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="p-2 bg-gray-50 rounded">
              <span className="font-medium">Patente Principal: </span>
              {mainPatent.id}
            </div>
            <div className="p-2 bg-gray-50 rounded">
              <span className="font-medium">Patente Citada: </span>
              {citedPatent.id}
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="font-medium">Similitud Semántica Global</h3>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div
                className="bg-blue-600 rounded-full h-4 transition-all duration-500"
                style={{ width: `${semanticData.similarity * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-600 text-center">
              {(semanticData.similarity * 100).toFixed(1)}% de similitud
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="font-medium mb-2">Términos Relevantes - Patente Principal</h3>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex flex-wrap gap-2">
                  {semanticData.main_terms.map((term, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 rounded transition-colors"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${term.score})`,
                        color: term.score > 0.5 ? 'white' : 'black'
                      }}
                    >
                      {term.token}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-medium mb-2">Términos Relevantes - Patente Citada</h3>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex flex-wrap gap-2">
                  {semanticData.cited_terms.map((term, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 rounded transition-colors"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${term.score})`,
                        color: term.score > 0.5 ? 'white' : 'black'
                      }}
                    >
                      {term.token}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Estado inicial cuando no hay datos ni patente seleccionada
  return (
    <div className="flex items-center justify-center h-full">
      <p className="text-gray-500">Seleccione una patente citada para ver la comparación</p>
    </div>
  );
};

if (typeof window !== 'undefined') {
  window.SemanticComparisonView = SemanticComparisonView;
}