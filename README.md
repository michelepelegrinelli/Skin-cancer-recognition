# Skin-cancer-recognition

Si propone lo sviluppo di un modello di machine learning in grado di analizzare immagini relative ai tumori della pelle al fine di determinare la loro classe in maligna o benigna.

## Fase 1: Informarsi sulle tipologie di tumore della pelle

In questa fase, ci informeremo sulle diverse tipologie di tumori della pelle per comprendere meglio il contesto specifico e le caratteristiche delle immagini da analizzare.

In allegato un approfondimento della natura dei tumori della pelle

## Fase 2: Raccolta dei dati

Raccoglieremo i dati da fonti attendibili e sicure. Per il nostro progetto, abbiamo utilizzato il dataset della International Skin Imaging Collaboration (ISIC) disponibile su Kaggle. Questo dataset contiene immagini etichettate di tumori della pelle suddivise in tumori maligni e benigni.

## Fase 3: Creazione e addestramento del modello di machine learning

Utilizzeremo le librerie TensorFlow e Keras per creare il nostro modello di machine learning. Sfrutteremo l'architettura MobileNetV2 come estrattore di caratteristiche per analizzare le immagini dei tumori della pelle.

Il dataset sarà suddiviso in due cartelle, una contenente le immagini di tumori maligni e una contenente le immagini di tumori benigni. Creeremo un dataset di training e un dataset di validazione per addestrare il modello.

Successivamente, addestreremo il modello utilizzando il dataset di training, eseguendo più epoche di addestramento. Ad esempio, abbiamo utilizzato 15 epoche per addestrare il nostro modello.

Dopo l'addestramento, valideremo il modello utilizzando il dataset di test e analizzeremo le metriche risultanti, come la perdita (loss) e l'accuratezza (accuracy), per valutare le prestazioni del modello.

Infine, testeremo il modello su immagini esterne al dataset per verificare la sua capacità di generalizzare e fare previsioni accurate su nuovi casi.

## Fase 4: Creazione dell'applicazione (prototipo)

Come passo finale, creeremo un'applicazione per fornire un'interfaccia utente intuitiva e interattiva per l'utilizzo del modello di riconoscimento dei tumori della pelle. Questo prototipo dell'applicazione consentirà agli utenti di caricare un'immagine di un tumore della pelle e ottenere una previsione sulla sua natura maligna o benigna.

Collaboratori:
- Pelegrinelli Michele
- Valenti Stefano
- Zampou Noufou
- Negrini Francesco
