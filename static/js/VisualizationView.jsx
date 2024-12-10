//import SemanticComparisonView from './static/js/SemanticComparisonView';

const Plot = React.memo(props => {
  const plotRef = React.useRef();

  React.useEffect(() => {
    if (plotRef.current && props.data) {
      window.Plotly.newPlot(
        plotRef.current,
        props.data,
        props.layout,
        props.config
      );
    }
  }, [props.data, props.layout]);

  return <div ref={plotRef} style={{ width: '100%', height: '100%' }} />;
});

const VisualizationView = ({ embeddings, updateVisualization }) => {
  const [plotType, setPlotType] = React.useState('cosine');
  const [searchTerm, setSearchTerm] = React.useState('');
  const [plotData, setPlotData] = React.useState(null);
  const [selectedPatentId, setSelectedPatentId] = React.useState(null);
  const [selectedPatent, setSelectedPatent] = React.useState(null);
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    // Verificar y loggear los datos cuando embeddings cambia
    if (embeddings) {
      console.log("Datos de embeddings recibidos:", embeddings);
      if (!embeddings.main_patent?.text) {
        console.error("La patente principal no tiene texto");
      }
    }
  }, [embeddings]);

  React.useEffect(() => {
    const loadPlotData = async () => {
      if (!embeddings) return;
      setIsLoading(true);
      /* setError(null); Produce error!!! */

      try {
        if (plotType === 'semantic' || plotType === 'bert') {
          if (!selectedPatent) {
            setIsLoading(false);
            return;
          }
          await loadSemanticData(selectedPatent);
        } else {
          const response = await fetch(`/api/visualization/${plotType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(embeddings)
          });

          if (!response.ok) throw new Error('Error al cargar visualización');
          const result = await response.json();
          setPlotData(JSON.parse(result));
        }
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadPlotData();
  }, [plotType, embeddings, selectedPatent]);

  const loadSemanticData = async (citedPatent) => {
    if (!embeddings?.main_patent?.text || !citedPatent?.text) {
      console.error('Faltan textos necesarios:', {
        mainText: !!embeddings?.main_patent?.text,
        citedText: !!citedPatent?.text
      });
      return;
    }

    try {
      const requestData = {
        main_text: embeddings.main_patent.text,
        cited_text: citedPatent.text
      };

      console.log('Enviando datos para análisis semántico:', requestData);

      const response = await fetch('/api/visualization/semantic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error en respuesta del servidor:', errorText);
        throw new Error(errorText || 'Error en el análisis semántico');
      }

      const result = await response.json();
      console.log('Resultado del análisis semántico:', result);
      setPlotData(result);
    } catch (error) {
      console.error('Error en análisis semántico:', error);
    }
  };

  const handlePatentSelect = (id) => {
    const selected = embeddings.cited_patents.find(p => p.id === id);
    console.log("Patente seleccionada:", selected);

    if (!selected?.text) {
      console.error('La patente seleccionada no tiene texto disponible');
      return;
    }

    setSelectedPatentId(id);
    setSelectedPatent(selected);

    if (plotType === 'semantic') {
      loadSemanticData(selected);
    } else if (plotData?.data) {
      const updatedData = plotData.data.map(trace => {
        if (trace.text && trace.text[0].includes(id)) {
          return {
            ...trace,
            line: { ...trace.line, color: 'green', width: 4 }
          };
        }
        return trace;
      });
      setPlotData({ ...plotData, data: updatedData });
    }
  };

  const handlePlotTypeChange = (event) => {
    setPlotType(event.target.value);
    if (event.target.value !== 'semantic' && event.target.value !== 'bert') {
      updateVisualization(event.target.value);
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-500">Cargando visualización...</p>
        </div>
      );
    }

    
    if (plotType === 'semantic') {
      return (
        <SemanticComparisonView
          mainPatent={embeddings?.main_patent}
          citedPatent={selectedPatent}
        />
      );
    } 

    if (plotType === 'bert') {
      return (
        <BertCrossAttentionView
          mainPatent={embeddings?.main_patent}
          citedPatent={selectedPatent}
        />
      );
    }

    return plotData && (
      <Plot
        data={plotData.data}
        layout={{
          ...plotData.layout,
          autosize: true,
          width: '100%',
          height: '100%'
        }}
        config={{ responsive: true }}
      />
    );
  };

  return (
    <div className="flex h-full">
      <div className="w-1/4 p-4 border-r bg-gray-50">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Tipo de gráfico
        </label>
        <select
          value={plotType}
          onChange={handlePlotTypeChange}
          className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="cosine">Distancia Coseno</option>
          <option value="euclidean">Distancia Euclidiana</option>
          <option value="semantic">Relación semántica</option>
          <option value="bert">Relación semántica - BERT</option>
        </select>
      </div>

      <div className="w-1/2 p-4 border-r h-full">
        {renderContent()}
      </div>

      <div className="w-1/4 p-4 bg-gray-50">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Buscar Id
          </label>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
            placeholder="Buscar..."
          />
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Lista de antecedentes
          </label>
          <div className="border rounded max-h-[calc(100vh-300px)] overflow-y-auto bg-white">
            {embeddings?.cited_patents
              .filter(patent =>
                patent.id.toLowerCase().includes(searchTerm.toLowerCase())
              )
              .map((patent) => (
                <div
                  key={patent.id}
                  onClick={() => handlePatentSelect(patent.id)}
                  className={`p-2 cursor-pointer hover:bg-gray-100 ${selectedPatentId === patent.id ? 'bg-blue-100' : ''
                    }`}
                >
                  {patent.id}
                </div>
              ))
            }
          </div>
        </div>
      </div>
    </div>
  );
};

if (typeof window !== 'undefined') {
  window.VisualizationView = VisualizationView;
}