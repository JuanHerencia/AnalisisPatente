const PatentAnalysisSystem = () => {
    const [currentView, setCurrentView] = React.useState(1);
    const [mainPatent, setMainPatent] = React.useState({ id: '', text: '' });
    const [citedPatents, setCitedPatents] = React.useState([]);
    const [currentCitedIndex, setCurrentCitedIndex] = React.useState(0);
    const [embeddings, setEmbeddings] = React.useState(null);
    const [loadingEmbeddings, setLoadingEmbeddings] = React.useState(false);
    const [hasModifiedTexts, setHasModifiedTexts] = React.useState(false);
    const fileInputRef = React.useRef(null);
    const [windowDimensions, setWindowDimensions] = React.useState({
        width: window.innerWidth,
        height: window.innerHeight
    });
    const [error, setError] = React.useState(null);

    const views = {
        1: {
            title: "Reinvindicaciones y antecedentes",
            leftButton: "Cargar archivo de reinvindicaciones",
            rightButton: "Extraer embeddings >>",
            nextView: 2
        },
        2: {
            title: "Datos de embeddings",
            leftButton: "<< Cargar reinvindicaciones y antecedentes",
            rightButton: "Análisis visual >>",
            prevView: 1,
            nextView: 3
        },
        3: {
            title: "Análisis visual",
            leftButton: "<< Datos de embeddings",
            rightButton: "Búsqueda de novedad de reinvindicacion >>",
            prevView: 2,
            nextView: 4
        },
        4: {
            title: "Resultados de novedad de la reinvindicación",
            leftButton: "<< Análisis visual",
            rightButton: "Cargar otra reinvindicación >>",
            prevView: 3,
            nextView: 1
        }
    };

    React.useEffect(() => {
        const handleResize = () => {
            setWindowDimensions({
                width: window.innerWidth,
                height: window.innerHeight
            });
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    const mainPatentId = Object.keys(data).find(key => key !== 'cited_document_id');

                    if (mainPatentId) {
                        setMainPatent({
                            id: mainPatentId,
                            text: data[mainPatentId]
                        });

                        if (data.cited_document_id) {
                            const citedArray = Object.entries(data.cited_document_id).map(([id, text]) => ({
                                id,
                                text
                            }));
                            setCitedPatents(citedArray);
                            setCurrentCitedIndex(0);
                        }
                        setHasModifiedTexts(true);
                    }
                } catch (error) {
                    console.error('Error al procesar el archivo JSON:', error);
                    alert('Error al procesar el archivo JSON. Asegúrese de que el formato sea correcto.');
                }
            };
            reader.readAsText(file);
        }
    };

    const generateEmbeddings = async () => {
        if (!mainPatent.id || citedPatents.length === 0) {
            alert('Por favor, cargue los datos de las patentes primero');
            return;
        }

        setLoadingEmbeddings(true);

        try {
            const data = {
                [mainPatent.id]: mainPatent.text,
                cited_document_id: citedPatents.reduce((acc, patent) => {
                    if (!patent.text) {
                        console.error(`Patente ${patent.id} no tiene texto`);
                    }
                    acc[patent.id] = patent.text || '';
                    return acc;
                }, {})
            };

            console.log('Enviando datos para embeddings:', data);

            const response = await fetch('/generate_embeddings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }

            const result = await response.json();
            console.log('Embeddings recibidos:', result);

            if (!result.embeddings) {
                throw new Error('No se recibieron embeddings en la respuesta');
            }

            // Asegurarse de que los textos se mantengan en los embeddings
            result.embeddings.main_patent.text = mainPatent.text;
            result.embeddings.cited_patents.forEach(cited => {
                const originalPatent = citedPatents.find(p => p.id === cited.id);
                if (originalPatent) {
                    cited.text = originalPatent.text;
                }
            });

            setEmbeddings(result.embeddings);
            setHasModifiedTexts(false);
            setCurrentView(2);
        } catch (error) {
            console.error('Error:', error);
            alert('Error al generar embeddings: ' + error.message);
        } finally {
            setLoadingEmbeddings(false);
        }
    };

    const handleMainPatentChange = (field, value) => {
        setMainPatent(prev => ({
            ...prev,
            [field]: value
        }));
        setHasModifiedTexts(true);
    };

    const handleCitedPatentChange = (field, value) => {
        if (citedPatents.length > 0) {
            const updatedCited = [...citedPatents];
            updatedCited[currentCitedIndex] = {
                ...updatedCited[currentCitedIndex],
                [field]: value
            };
            setCitedPatents(updatedCited);
            setHasModifiedTexts(true);
        }
    };

    const navigateCited = (direction) => {
        if (direction === 'prev' && currentCitedIndex > 0) {
            setCurrentCitedIndex(prev => prev - 1);
        } else if (direction === 'next' && currentCitedIndex < citedPatents.length - 1) {
            setCurrentCitedIndex(prev => prev + 1);
        }
    };

    const downloadJSON = () => {
        const output = {
            [mainPatent.id]: mainPatent.text,
            cited_document_id: citedPatents.reduce((acc, patent) => {
                acc[patent.id] = patent.text;
                return acc;
            }, {})
        };

        const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'patent_data.json';
        a.click();
        URL.revokeObjectURL(url);
    };

    const renderEmbeddingsView = () => {
        if (!embeddings) {
            return (
                <div className="h-full flex items-center justify-center text-gray-500">
                    No hay embeddings generados
                </div>
            );
        }

        console.log('Renderizando embeddings:', embeddings); // Debug

        const formatVector = (vector) => {
            if (!Array.isArray(vector)) {
                console.error('Vector inválido:', vector);
                return 'Error: Vector inválido';
            }
            return vector.map((value, index) => {
                const formattedValue = Number(value).toFixed(6);
                return `[${index}] ${formattedValue}`;
            }).join('\n');
        };

        const formatReducedVector = (vector) => {
            if (!Array.isArray(vector)) {
                console.error('Vector reducido inválido:', vector);
                return 'Error: Vector inválido';
            }
            return `x: ${vector[0].toFixed(6)}\ny: ${vector[1].toFixed(6)}\nz: ${vector[2].toFixed(6)}`;
        };

        if (!embeddings.main_patent || !embeddings.cited_patents) {
            return (
                <div className="h-full flex items-center justify-center text-red-500">
                    Error: Estructura de embeddings inválida
                </div>
            );
        }

        return (
            <div className="h-full overflow-auto p-4">
                {embeddings.from_cache && (
                    <div className="mb-4 p-2 bg-blue-50 text-blue-700 rounded">
                        Usando embeddings previamente calculados
                    </div>
                )}

                {/* Patente Principal */}
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-medium">Embeddings de Patente Principal</h3>
                        <span className="text-sm text-gray-500">
                            Vector de dimensión: {embeddings.main_patent.embedding.length}
                        </span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                        <p className="font-medium mb-2">ID: {embeddings.main_patent.id}</p>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            <div>
                                <p className="text-sm font-medium text-gray-700 mb-2">Vector Original</p>
                                <pre className="text-sm bg-white p-4 rounded border overflow-auto max-h-96 font-mono">
                                    {formatVector(embeddings.main_patent.embedding)}
                                </pre>
                            </div>
                            <div>
                                <p className="text-sm font-medium text-gray-700 mb-2">Vector Reducido (t-SNE)</p>
                                <pre className="text-sm bg-white p-4 rounded border overflow-auto max-h-96 font-mono">
                                    {formatReducedVector(embeddings.main_patent.reduced_embedding)}
                                </pre>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Patentes Citadas */}
                <div>
                    <h3 className="text-lg font-medium mb-4">Embeddings de Patentes Citadas</h3>
                    <div className="space-y-6">
                        {embeddings.cited_patents.map((patent, index) => (
                            <div key={index} className="bg-gray-50 p-4 rounded">
                                <div className="flex items-center justify-between mb-2">
                                    <p className="font-medium">ID: {patent.id}</p>
                                    <span className="text-sm text-gray-500">
                                        Vector de dimensión: {patent.embedding.length}
                                    </span>
                                </div>
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                    <div>
                                        <p className="text-sm font-medium text-gray-700 mb-2">Vector Original</p>
                                        <pre className="text-sm bg-white p-4 rounded border overflow-auto max-h-96 font-mono">
                                            {formatVector(patent.embedding)}
                                        </pre>
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-700 mb-2">Vector Reducido (t-SNE)</p>
                                        <pre className="text-sm bg-white p-4 rounded border overflow-auto max-h-96 font-mono">
                                            {formatReducedVector(patent.reduced_embedding)}
                                        </pre>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    const renderMainContent = () => {
        if (currentView === 1) {
            return (
                <div className="view-content">
                    <div className="patent-sections flex flex-col md:flex-row h-full">
                        {/* Sección de Patente Principal */}
                        <div className="patent-section p-4 md:w-1/2 md:border-r">
                            <div className="h-full flex flex-col">
                                <h3 className="text-lg font-medium mb-4">Patente Principal</h3>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        ID de Patente
                                    </label>
                                    <input
                                        type="text"
                                        value={mainPatent.id}
                                        onChange={(e) => handleMainPatentChange('id', e.target.value)}
                                        className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                                    />
                                </div>
                                <div className="flex-grow">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Texto de Patente
                                    </label>
                                    <textarea
                                        value={mainPatent.text}
                                        onChange={(e) => handleMainPatentChange('text', e.target.value)}
                                        className="w-full h-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Sección de Patentes Citadas */}
                        <div className="patent-section p-4 md:w-1/2">
                            <div className="h-full flex flex-col">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg font-medium">
                                        Patentes Citadas ({citedPatents.length > 0 ? currentCitedIndex + 1 : 0} de {citedPatents.length})
                                    </h3>
                                    <div className="space-x-2">
                                        <button
                                            onClick={() => navigateCited('prev')}
                                            disabled={currentCitedIndex === 0}
                                            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
                                        >
                                            ←
                                        </button>
                                        <button
                                            onClick={() => navigateCited('next')}
                                            disabled={currentCitedIndex === citedPatents.length - 1}
                                            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
                                        >
                                            →
                                        </button>
                                    </div>
                                </div>

                                {citedPatents.length > 0 ? (
                                    <div className="flex-grow flex flex-col">
                                        <div className="mb-4">
                                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                                ID de Patente Citada
                                            </label>
                                            <input
                                                type="text"
                                                value={citedPatents[currentCitedIndex]?.id || ''}
                                                onChange={(e) => handleCitedPatentChange('id', e.target.value)}
                                                className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                                            />
                                        </div>
                                        <div className="flex-grow">
                                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                                Texto de Patente Citada
                                            </label>
                                            <textarea
                                                value={citedPatents[currentCitedIndex]?.text || ''}
                                                onChange={(e) => handleCitedPatentChange('text', e.target.value)}
                                                className="w-full h-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex-grow flex items-center justify-center text-gray-500">
                                        No hay patentes citadas cargadas
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Barra de Acciones */}
                    <div className="action-bar p-4 bg-gray-50 border-t mt-auto">
                        <div className="flex justify-between">
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileUpload}
                                accept=".json"
                                className="hidden"
                            />
                            <button
                                onClick={() => fileInputRef.current.click()}
                                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                Cargar archivo
                            </button>
                            <button
                                onClick={downloadJSON}
                                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                                disabled={!mainPatent.id}
                            >
                                Guardar cambios
                            </button>
                        </div>
                    </div>
                </div>
            );
        } else if (currentView === 2) {
            return renderEmbeddingsView();
        } else if (currentView === 3 && window.VisualizationView) {
            return React.createElement(window.VisualizationView, {
                embeddings: embeddings,
                updateVisualization: async (type) => {
                    try {
                        const response = await fetch(`/api/visualization/${type}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(embeddings)
                        });
                        if (!response.ok) throw new Error('Error en visualización');
                        const result = await response.json();
                        console.log('Visualización actualizada:', result);
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }
            });
        }
        return (
            <div className="h-full flex items-center justify-center">
                Contenido de la Vista {currentView}
            </div>
        );
    };

    const handleNewClaim = async () => {
        try {
            const response = await fetch('/clear_session', {
                method: 'POST'
            });

            if (response.ok) {
                window.location.href = '/reset_view';
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div className="main-container">
            {/* Header */}
            <header className="bg-white shadow">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
                        Sistema de Análisis de Reinvindicaciones
                    </h1>
                    <h2 className="text-lg md:text-xl text-gray-600">Visualización y Análisis</h2>
                    <p className="text-sm md:text-base text-gray-500">Herramienta para análisis de las reinvindicaciones y sus antecedentes</p>
                </div>
            </header>

            {/* Content Wrapper */}
            <div className="content-wrapper">
                <main className="flex-grow p-4">
                    <div className="bg-white shadow rounded-lg h-full flex flex-col">
                        <h2 className="text-xl md:text-2xl font-bold text-center p-4 border-b">
                            {views[currentView].title}
                        </h2>

                        {/* Marco principal */}
                        <div className="flex-grow overflow-hidden">
                            {renderMainContent()}
                        </div>

                        {/* Navegación */}
                        <div className="p-4 border-t">
                            <div className="flex justify-between">
                                {currentView > 1 && (
                                    <button
                                        onClick={() => setCurrentView(views[currentView].prevView)}
                                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                                    >
                                        {views[currentView].leftButton}
                                    </button>
                                )}


                                {currentView === 4 ? (
                                    <button
                                        onClick={handleNewClaim}
                                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                                    >
                                        {views[currentView].rightButton}
                                    </button>
                                ) : (
                                    <button
                                        onClick={() => {
                                            if (currentView === 1 && !loadingEmbeddings) {
                                                generateEmbeddings();
                                            } else {
                                                setCurrentView(views[currentView].nextView);
                                            }
                                        }}
                                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                                        disabled={currentView === 1 && loadingEmbeddings}
                                    >
                                        {currentView === 1 ?
                                            (loadingEmbeddings ? 'Generando...' : views[currentView].rightButton)
                                            : views[currentView].rightButton}
                                    </button>
                                )}



                            </div>
                        </div>
                    </div>
                </main>
            </div>

            {/* Footer */}
            <footer className="bg-white shadow mt-auto">
                <div className="max-w-7xl mx-auto p-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-gray-50 p-4 rounded">
                            <h3 className="font-bold mb-2">Determinación de la novedad en reinvindicaciones</h3>
                        </div>
                        <div className="bg-gray-50 p-4 rounded">
                            <h3 className="font-bold mb-2">Autores</h3>
                            <p>José Zúñiga - Jeyson Lino - Juan Herencia</p>
                        </div>
                        <div className="bg-gray-50 p-4 rounded">
                            <h3 className="font-bold mb-2">Universidad de Ingeniería y Tecnología - UTEC</h3>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

// Renderizar el componente
//const root = ReactDOM.createRoot(document.getElementById('root'));
//root.render(<PatentAnalysisSystem />);

// Modificar el renderizado para usar createRoot correctamente
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(React.createElement(PatentAnalysisSystem));