export type Locale = "en" | "de";

export type Messages = {
  common: {
    loading: string;
    save: string;
    saving: string;
    cancel: string;
    close: string;
    export: string;
    refresh: string;
    error: string;
  };
  header: {
    metrics: string;
      memory: string;
    settings: string;
    openNavigation: string;
    activeSession: string;
    exportConversation: string;
    signOut: string;
    signingOut: string;
    justNow: string;
    minutesAgo: string;
    hoursAgo: string;
    daysAgo: string;
    messageCountOne: string;
    messageCountOther: string;
  };
  sidebar: {
    platformSubtitle: string;
    closeNavigation: string;
    startNewChat: string;
    newChat: string;
    exitBulkMode: string;
    selectMultiple: string;
    showArchived: string;
    hideArchived: string;
    archived: string;
    searchConversations: string;
    clearSearch: string;
    searching: string;
    results: string;
    noResultsFor: string;
    selectedCount: string;
    selectAll: string;
    archiveSelected: string;
    deleteSelected: string;
    deleteSelectedConfirm: string;
    chatHistory: string;
    pinned: string;
    recentChats: string;
    noConversations: string;
    settingsForUser: string;
    settings: string;
    moreOptions: string;
    rename: string;
    pin: string;
    unpin: string;
    archive: string;
    unarchive: string;
    exportJson: string;
    exportMarkdown: string;
    delete: string;
    deleteConversationConfirm: string;
    cancelSelection: string;
    select: string;
  };
  login: {
    loginFailed: string;
    developmentMode: string;
    authDisabled: string;
    authDisabledDescription: string;
    enterApplication: string;
    login: string;
    welcomeTo: string;
    signInAsRole: string;
    username: string;
    password: string;
    enterUsername: string;
    enterPassword: string;
    signingIn: string;
    signIn: string;
    platformFallback: string;
    applicationFallback: string;
  };
  settingsPage: {
    title: string;
    subtitleFallback: string;
    appSubtitle: string;
    active: string;
    debugCurrentSettings: string;
    lastUpdated: string;
  };
  settingsPanel: {
    language: string;
    toolSettings: string;
    loadingTools: string;
    ragConfiguration: string;
    loadingDefaults: string;
    knowledgeCollection: string;
    topKResults: string;
    similarityThreshold: string;
    modelSelection: string;
    systemDefault: string;
    loadingModels: string;
    noModelsAvailable: string;
    saveSettings: string;
    settingsSaved: string;
    failedToSaveSettings: string;
    reingestSectionTitle: string;
    reingestDescription: string;
    reingestAllDocuments: string;
    reingesting: string;
    confirmReingestAll: string;
    reingestSuccess: string;
    reingestFailed: string;
  };
  chat: {
    noMessagesYet: string;
    startConversationHint: string;
    aiAgent: string;
    aiThinking: string;
    excludeFromNextMessage: string;
    includeInNextMessage: string;
    deleteAttachment: string;
    uploadingAttachment: string;
    message: string;
    typeYourMessage: string;
    sendMessage: string;
    send: string;
    attachmentCountOne: string;
    attachmentCountOther: string;
    toggleWorkspaceSidebar: string;
    workspace: string;
    newConversation: string;
    copyMessage: string;
    regenerateResponse: string;
    responseMetrics: string;
    responseTime: string;
    inputTokens: string;
    outputTokens: string;
    finishReason: string;
    model: string;
    temperature: string;
    toolsInvoked: string;
    feedbackSaved: string;
    feedbackRemoved: string;
    workspaceDocumentSaved: string;
    messageFailed: string;
    failedToSendMessage: string;
    couldNotCreateConversationForAttachment: string;
    attachmentUploadFailed: string;
    attachmentUploaded: string;
    attachmentDeleteFailed: string;
    attachmentRemoved: string;
    templates: {
      explainConcept: string;
      helpCode: string;
      summarizeText: string;
      translate: string;
      brainstormIdeas: string;
      debugIssue: string;
      explainPrompt: string;
      codePrompt: string;
      summarizePrompt: string;
      translatePrompt: string;
      brainstormPrompt: string;
      debugPrompt: string;
    };
  };
  workspace: {
    loadFailed: string;
    saveFailed: string;
    deleteFailed: string;
    downloaded: string;
    reattachFailed: string;
    attachedToChat: string;
    commandAdded: string;
    uploadFailed: string;
    documentUploaded: string;
    toggleSidebar: string;
    closeSidebar: string;
    uploadDocument: string;
    uploading: string;
    myDocuments: string;
    edit: string;
    reference: string;
    download: string;
    delete: string;
    editor: string;
    closeEditor: string;
    resizeEditor: string;
    editDocumentPlaceholder: string;
    saveChanges: string;
    attachToChat: string;
    noDocuments: string;
    uploadToGetStarted: string;
    noActiveConversation: string;
    loadDocumentFailed: string;
    updateDocumentFailed: string;
    deleteDocumentFailed: string;
    saveChangesFailed: string;
    uploaded: string;
    commandReferenceTemplate: string;
  };
  exportDialog: {
    exportFailedNoData: string;
    exportFailedTryAgain: string;
    exportedSuccessfully: string;
    exportFailed: string;
    title: string;
    closeDialog: string;
    format: string;
    markdown: string;
    markdownDescription: string;
    json: string;
    jsonDescription: string;
    exporting: string;
  };
  metrics: {
    title: string;
    subtitle: string;
    guardrailHealthyTitle: string;
    guardrailWarningTitle: string;
    guardrailHealthyHint: string;
    guardrailWarningHint: string;
    sectionsLabel: string;
    realtimeTab: string;
    trendsTab: string;
    realtimeTitle: string;
    trendsTitle: string;
    loadingMetrics: string;
    loadingTrends: string;
    savePreferences: string;
    preferencesSaved: string;
    preferencesFailed: string;
    dateRange: string;
    invalidRange: string;
    last7Days: string;
    last30Days: string;
    last90Days: string;
    autoRefresh60s: string;
    lastUpdated: string;
    export: string;
    totalRequests: string;
    totalTokens: string;
    avgLatency: string;
    activeSessions: string;
    attachmentsInjected: string;
    requestsWithoutAttachments: string;
    attachmentRetryTriggers: string;
    attachmentRetrySuccess: string;
    profileIdMismatches: string;
    memoryOperations: string;
    llmCallsByProvider: string;
    providerModelStatus: string;
    count: string;
    attachmentContextMetrics: string;
    attachmentRetryGuardrail: string;
    usageTrends: string;
    dailyRequestsAndUsers: string;
    requests: string;
    users: string;
    tokenConsumption: string;
    totalAndAverageTokens: string;
    total: string;
    average: string;
    latencyAnalysis: string;
    minAverageMaxDuration: string;
    min: string;
    max: string;
    memoryTrends: string;
    dailyMemoryOpsQualityAndErrorRate: string;
    noMemoryTrendData: string;
    summary: string;
    keyMetricsForPeriod: string;
    uniqueUsers: string;
    memoryLoads: string;
    memoryUpdates: string;
    memoryQueries: string;
    memoryErrorRate: string;
    totalOps: string;
    avgQuality: string;
    coOccurrenceNodes: string;
    coOccurrenceEdges: string;
    rankingLatency: string;
    operation: string;
    success: string;
    error: string;
    noUsageData: string;
    noTokenData: string;
    noLatencyData: string;
    noSummaryData: string;
    selectAtLeastOneMetric: string;
    errorLoadingTrends: string;
    days: string;
    na: string;
    retrievalFeedback: string;
    retrievalFeedbackSubtitle: string;
    retrievalSignalsTotal: string;
    retrievedWithRating: string;
    retrievedWithoutRating: string;
    priorRatingCoverage: string;
    ratedAfterRetrieval: string;
    feedbackCoverage: string;
    ratingBuckets: string;
    bucketNone: string;
    bucketLow: string;
    bucketMid: string;
    bucketHigh: string;
    noRetrievalData: string;
  };
  memories: {
    title: string;
    subtitle: string;
    refresh: string;
    total: string;
    recent7d: string;
    rated: string;
    unrated: string;
    avgImportance: string;
    coverage: string;
    filterPlaceholder: string;
    memory: string;
    importance: string;
    rating: string;
    saved: string;
    loading: string;
    noMemoriesMatchFilter: string;
    noMemoriesYet: string;
    related: string;
    deleteMemory: string;
    confirmYes: string;
    confirmNo: string;
    pageOfTotal: string;
    previous: string;
    next: string;
    rateStars: string;
  };
  charts: {
    loading: string;
    noData: string;
    selectAtLeastOneSeries: string;
    dailyUsageTrends: string;
    dailyTokenConsumption: string;
    requestLatencyAnalysisMs: string;
    requestsByStatus: string;
    noRequestData: string;
    tokenUsage: string;
    noTokenUsageData: string;
    tokenUsageByModel: string;
    total: string;
    tokens: string;
    loads: string;
    updates: string;
    errors: string;
    errorRatePercent: string;
    qualityPercent: string;
    loadingMemoryTrends: string;
    noMemoryTrendData: string;
    maxLatency: string;
    avgLatency: string;
    minLatency: string;
  };
};

export const messages: Record<Locale, Messages> = {
  en: {
    common: {
      loading: "Loading...",
      save: "Save",
      saving: "Saving...",
      cancel: "Cancel",
      close: "Close",
      export: "Export",
      refresh: "Refresh",
      error: "Error",
    },
    header: {
      metrics: "Metrics",
      memory: "Memory",
      settings: "Settings",
      openNavigation: "Open navigation",
      activeSession: "Active session",
      exportConversation: "Export conversation",
      signOut: "Sign out",
      signingOut: "Signing out...",
      justNow: "just now",
      minutesAgo: "{count}m ago",
      hoursAgo: "{count}h ago",
      daysAgo: "{count}d ago",
      messageCountOne: "{count} msg",
      messageCountOther: "{count} msgs",
    },
    sidebar: {
      platformSubtitle: "Universal Agentic Orchestration Platform",
      closeNavigation: "Close navigation",
      startNewChat: "Start a new chat",
      newChat: "New Chat",
      exitBulkMode: "Exit bulk mode",
      selectMultiple: "Select multiple",
      showArchived: "Show archived",
      hideArchived: "Hide archived",
      archived: "Archived",
      searchConversations: "Search conversations...",
      clearSearch: "Clear search",
      searching: "Searching...",
      results: "Results ({count})",
      noResultsFor: "No results for \"{query}\"",
      selectedCount: "{count} selected",
      selectAll: "All",
      archiveSelected: "Archive selected",
      deleteSelected: "Delete selected",
      deleteSelectedConfirm: "Delete {count} conversation(s)?",
      chatHistory: "Chat history",
      pinned: "Pinned",
      recentChats: "Recent Chats",
      noConversations: "No conversations yet",
      settingsForUser: "Open settings for {name}",
      settings: "Settings",
      moreOptions: "More options",
      rename: "Rename",
      pin: "Pin",
      unpin: "Unpin",
      archive: "Archive",
      unarchive: "Unarchive",
      exportJson: "Export JSON",
      exportMarkdown: "Export Markdown",
      delete: "Delete",
      deleteConversationConfirm: "Delete this conversation?",
      cancelSelection: "Cancel",
      select: "Select",
    },
    login: {
      loginFailed: "Login failed",
      developmentMode: "Development Mode",
      authDisabled: "Authentication is currently disabled.",
      authDisabledDescription:
        "This environment bypasses the login screen. Set AUTH_ENABLED=true and configure your auth secrets to require sign-in.",
      enterApplication: "Enter {app}",
      login: "Login",
      welcomeTo: "Welcome to {app}",
      signInAsRole: "Sign in as {role} using the operator account configured in the environment variables.",
      username: "Username",
      password: "Password",
      enterUsername: "Enter your username",
      enterPassword: "Enter your password",
      signingIn: "Signing in...",
      signIn: "Sign in",
      platformFallback: "the platform",
      applicationFallback: "application",
    },
    settingsPage: {
      title: "Settings",
      subtitleFallback: "Account and system preferences in one place",
      appSubtitle: "{app} settings",
      active: "active",
      debugCurrentSettings: "Debug: Current Settings",
      lastUpdated: "Last updated: {value}",
    },
    settingsPanel: {
      language: "Language",
      toolSettings: "Tool Settings",
      loadingTools: "Loading tools...",
      ragConfiguration: "RAG Configuration",
      loadingDefaults: "Loading defaults...",
      knowledgeCollection: "Knowledge Collection (System default: {value})",
      topKResults: "Top-K Results (System default: {default}): {value}",
      similarityThreshold: "Similarity Threshold",
      modelSelection: "Model Selection",
      systemDefault: "System default: {value}",
      loadingModels: "Loading models...",
      noModelsAvailable: "No models available - is Ollama running?",
      saveSettings: "Save Settings",
      settingsSaved: "Settings saved",
      failedToSaveSettings: "Failed to save settings",
      reingestSectionTitle: "Knowledge Re-ingestion",
      reingestDescription: "Trigger a full re-index of all currently available documents in the ingestion source.",
      reingestAllDocuments: "Re-ingest all documents",
      reingesting: "Re-ingesting...",
      confirmReingestAll: "This will clear and re-index the configured collection. Continue?",
      reingestSuccess: "Re-ingestion completed: {processed} files, {chunks} chunks",
      reingestFailed: "Failed to re-ingest documents",
    },
    chat: {
      noMessagesYet: "No messages yet",
      startConversationHint: "Start a conversation or pick a template below",
      aiAgent: "AI Agent",
      aiThinking: "AI is thinking",
      excludeFromNextMessage: "Exclude from next message",
      includeInNextMessage: "Include in next message",
      deleteAttachment: "Delete attachment",
      uploadingAttachment: "Uploading attachment...",
      message: "Message",
      typeYourMessage: "Type your message...",
      sendMessage: "Send message",
      send: "Send",
      attachmentCountOne: "{count} attachment selected",
      attachmentCountOther: "{count} attachments selected",
      toggleWorkspaceSidebar: "Toggle workspace sidebar",
      workspace: "Workspace",
      newConversation: "New conversation",
      copyMessage: "Copy message",
      regenerateResponse: "Regenerate this response",
      responseMetrics: "Response metrics",
      responseTime: "Response time",
      inputTokens: "Input tokens",
      outputTokens: "Output tokens",
      finishReason: "Finish reason",
      model: "Model",
      temperature: "Temperature",
      toolsInvoked: "Tools invoked",
      feedbackSaved: "Feedback saved",
      feedbackRemoved: "Feedback removed",
      workspaceDocumentSaved: "Workspace document saved",
      messageFailed: "Message failed",
      failedToSendMessage: "Failed to send message",
      couldNotCreateConversationForAttachment: "Could not create conversation for attachment",
      attachmentUploadFailed: "Attachment upload failed",
      attachmentUploaded: "Attachment uploaded",
      attachmentDeleteFailed: "Attachment delete failed",
      attachmentRemoved: "Attachment removed",
      templates: {
        explainConcept: "Explain a concept",
        helpCode: "Help me code",
        summarizeText: "Summarize text",
        translate: "Translate",
        brainstormIdeas: "Brainstorm ideas",
        debugIssue: "Debug an issue",
        explainPrompt: "Explain how ",
        codePrompt: "Write a function that ",
        summarizePrompt: "Summarize the following text:\n\n",
        translatePrompt: "Translate the following to English:\n\n",
        brainstormPrompt: "Give me creative ideas for ",
        debugPrompt: "Help me debug this issue:\n\n",
      },
    },
    workspace: {
      loadFailed: "Load failed",
      saveFailed: "Save failed",
      deleteFailed: "Delete failed",
      downloaded: "Downloaded",
      reattachFailed: "Re-attach failed",
      attachedToChat: "Attached to chat",
      commandAdded: "Command added to composer",
      uploadFailed: "Upload failed",
      documentUploaded: "Document uploaded",
      toggleSidebar: "Toggle workspace documents sidebar",
      closeSidebar: "Close workspace sidebar",
      uploadDocument: "Upload Document",
      uploading: "Uploading...",
      myDocuments: "My Documents ({count})",
      edit: "Edit",
      reference: "Reference",
      download: "Download",
      delete: "Delete",
      editor: "Editor",
      closeEditor: "Close editor",
      resizeEditor: "Drag to resize editor",
      editDocumentPlaceholder: "Edit document content...",
      saveChanges: "Save Changes",
      attachToChat: "Attach to Chat",
      noDocuments: "No documents yet",
      uploadToGetStarted: "Upload or create documents to get started",
      noActiveConversation: "No active conversation.",
      loadDocumentFailed: "Failed to load document",
      updateDocumentFailed: "Failed to update document",
      deleteDocumentFailed: "Failed to delete document",
      saveChangesFailed: "Failed to save changes",
      uploaded: "Uploaded",
      commandReferenceTemplate:
        "Reference workspace document \"{filename}\" (id: {id}) in your analysis.",
    },
    exportDialog: {
      exportFailedNoData: "Export failed - no data returned.",
      exportFailedTryAgain: "Export failed. Please try again.",
      exportedSuccessfully: "Exported successfully",
      exportFailed: "Export failed",
      title: "Export Conversation",
      closeDialog: "Close dialog",
      format: "Format",
      markdown: "Markdown",
      markdownDescription: "Human-readable .md file",
      json: "JSON",
      jsonDescription: "Structured data with metadata",
      exporting: "Exporting...",
    },
    metrics: {
      title: "{app} Metrics",
      subtitle: "System performance and analytics for {role}",
      guardrailHealthyTitle: "Profile Guardrail Healthy",
      guardrailWarningTitle: "Profile Guardrail Warning",
      guardrailHealthyHint: "No profile mismatch events observed",
      guardrailWarningHint: "Profile mismatch events detected",
      sectionsLabel: "Metrics sections",
      realtimeTab: "Real-Time Metrics",
      trendsTab: "Analytics Trends",
      realtimeTitle: "Real-Time Metrics",
      trendsTitle: "Analytics Trends",
      loadingMetrics: "Loading metrics...",
      loadingTrends: "Loading trends...",
      savePreferences: "Saving preferences...",
      preferencesSaved: "Preferences saved",
      preferencesFailed: "Failed to save",
      dateRange: "Date Range",
      invalidRange: "Invalid range",
      last7Days: "Last 7 days",
      last30Days: "Last 30 days",
      last90Days: "Last 90 days",
      autoRefresh60s: "Auto-refresh (60s)",
      lastUpdated: "Last updated",
      export: "Export",
      totalRequests: "Total Requests",
      totalTokens: "Total Tokens",
      avgLatency: "Avg Latency",
      activeSessions: "Active Sessions",
      attachmentsInjected: "Attachments Injected",
      requestsWithoutAttachments: "Requests Without Attachments",
      attachmentRetryTriggers: "Attachment Retry Triggers",
      attachmentRetrySuccess: "Attachment Retry Success",
      profileIdMismatches: "Profile ID Mismatches",
      memoryOperations: "Memory Operations",
      llmCallsByProvider: "LLM Calls by Provider",
      providerModelStatus: "Provider/Model/Status",
      count: "Count",
      attachmentContextMetrics: "Attachment Context Metrics",
      attachmentRetryGuardrail: "Attachment Retry Guardrail",
      usageTrends: "Usage Trends",
      dailyRequestsAndUsers: "Daily requests and unique users",
      requests: "Requests",
      users: "Users",
      tokenConsumption: "Token Consumption",
      totalAndAverageTokens: "Total and average tokens per day",
      total: "Total",
      average: "Average",
      latencyAnalysis: "Latency Analysis",
      minAverageMaxDuration: "Min, average, and maximum request duration",
      min: "Min",
      max: "Max",
      memoryTrends: "Memory Trends",
      dailyMemoryOpsQualityAndErrorRate: "Daily loads, updates, errors, quality, and error rate",
      noMemoryTrendData: "No memory trend data",
      summary: "Summary",
      keyMetricsForPeriod: "Key metrics for the selected period",
      uniqueUsers: "Unique Users",
      memoryLoads: "Memory Loads",
      memoryUpdates: "Memory Updates",
      memoryQueries: "Memory Queries",
      memoryErrorRate: "Memory Error Rate",
      totalOps: "Total Ops",
      avgQuality: "Avg Quality",
      coOccurrenceNodes: "Co-Occ Nodes",
      coOccurrenceEdges: "Co-Occ Edges",
      rankingLatency: "Ranking Latency",
      operation: "Operation",
      success: "Success",
      error: "Error",
      noUsageData: "No usage data available",
      noTokenData: "No token data available",
      noLatencyData: "No latency data available",
      noSummaryData: "No summary data available",
      selectAtLeastOneMetric: "Select at least one metric",
      errorLoadingTrends: "Error loading trends",
      days: "days",
      na: "N/A",
      retrievalFeedback: "Retrieval Feedback Loop",
      retrievalFeedbackSubtitle: "How often retrieved memories are subsequently rated by the user",
      retrievalSignalsTotal: "Retrieval Signals",
      retrievedWithRating: "Retrieved with Rating",
      retrievedWithoutRating: "Retrieved without Rating",
      priorRatingCoverage: "Prior Rating Coverage",
      ratedAfterRetrieval: "Rated after Retrieval",
      feedbackCoverage: "Feedback Coverage",
      ratingBuckets: "Rating Buckets at Retrieval",
      bucketNone: "Unrated",
      bucketLow: "Low (1–2)",
      bucketMid: "Mid (3)",
      bucketHigh: "High (4–5)",
      noRetrievalData: "No retrieval quality data yet",
    },
    memories: {
      title: "Memory",
      subtitle: "Memories the agent has learned about you",
      refresh: "Refresh",
      total: "Total",
      recent7d: "Recent 7d",
      rated: "Rated",
      unrated: "Unrated",
      avgImportance: "Avg importance",
      coverage: "{count}% coverage",
      filterPlaceholder: "Filter memories...",
      memory: "Memory",
      importance: "Importance",
      rating: "Rating",
      saved: "Saved",
      loading: "Loading...",
      noMemoriesMatchFilter: "No memories match your filter.",
      noMemoriesYet: "No memories yet.",
      related: "related",
      deleteMemory: "Delete memory",
      confirmYes: "Yes",
      confirmNo: "No",
      pageOfTotal: "Page {page} of {pages} ({total} total)",
      previous: "Previous",
      next: "Next",
      rateStars: "Rate {count} stars",
    },
    charts: {
      loading: "Loading...",
      noData: "No data available",
      selectAtLeastOneSeries: "Select at least one series",
      dailyUsageTrends: "Daily Usage Trends",
      dailyTokenConsumption: "Daily Token Consumption",
      requestLatencyAnalysisMs: "Request Latency Analysis (ms)",
      requestsByStatus: "Requests by Status",
      noRequestData: "No request data available",
      tokenUsage: "Token Usage",
      noTokenUsageData: "No token usage data available",
      tokenUsageByModel: "Token Usage by Model",
      total: "Total",
      tokens: "Tokens",
      loads: "Loads",
      updates: "Updates",
      errors: "Errors",
      errorRatePercent: "Error Rate %",
      qualityPercent: "Quality %",
      loadingMemoryTrends: "Loading memory trends...",
      noMemoryTrendData: "No memory trend data",
      maxLatency: "Max Latency",
      avgLatency: "Avg Latency",
      minLatency: "Min Latency",
    },
  },
  de: {
    common: {
      loading: "Wird geladen...",
      save: "Speichern",
      saving: "Wird gespeichert...",
      cancel: "Abbrechen",
      close: "Schließen",
      export: "Exportieren",
      refresh: "Aktualisieren",
      error: "Fehler",
    },
    header: {
      metrics: "Metrik",
      memory: "Speicher",
      settings: "Einstellungen",
      openNavigation: "Navigation öffnen",
      activeSession: "Aktive Sitzung",
      exportConversation: "Unterhaltung exportieren",
      signOut: "Abmelden",
      signingOut: "Melde ab...",
      justNow: "gerade eben",
      minutesAgo: "vor {count} Min.",
      hoursAgo: "vor {count} Std.",
      daysAgo: "vor {count} T.",
      messageCountOne: "{count} Nachricht",
      messageCountOther: "{count} Nachrichten",
    },
    sidebar: {
      platformSubtitle: "Universelle Agenten-Orchestrierungsplattform",
      closeNavigation: "Navigation schließen",
      startNewChat: "Neuen Chat starten",
      newChat: "Neuer Chat",
      exitBulkMode: "Mehrfachauswahl beenden",
      selectMultiple: "Mehrfach auswählen",
      showArchived: "Archivierte anzeigen",
      hideArchived: "Archivierte ausblenden",
      archived: "Archiviert",
      searchConversations: "Unterhaltungen durchsuchen...",
      clearSearch: "Suche löschen",
      searching: "Suche läuft...",
      results: "Ergebnisse ({count})",
      noResultsFor: "Keine Treffer für \"{query}\"",
      selectedCount: "{count} ausgewählt",
      selectAll: "Alle",
      archiveSelected: "Ausgewählte archivieren",
      deleteSelected: "Ausgewählte löschen",
      deleteSelectedConfirm: "{count} Unterhaltung(en) löschen?",
      chatHistory: "Chat-Verlauf",
      pinned: "Angeheftet",
      recentChats: "Letzte Chats",
      noConversations: "Noch keine Unterhaltungen",
      settingsForUser: "Einstellungen für {name} öffnen",
      settings: "Einstellungen",
      moreOptions: "Weitere Optionen",
      rename: "Umbenennen",
      pin: "Anheften",
      unpin: "Lösen",
      archive: "Archivieren",
      unarchive: "Wiederherstellen",
      exportJson: "JSON exportieren",
      exportMarkdown: "Markdown exportieren",
      delete: "Löschen",
      deleteConversationConfirm: "Diese Unterhaltung löschen?",
      cancelSelection: "Abbrechen",
      select: "Auswählen",
    },
    login: {
      loginFailed: "Anmeldung fehlgeschlagen",
      developmentMode: "Entwicklungsmodus",
      authDisabled: "Authentifizierung ist derzeit deaktiviert.",
      authDisabledDescription:
        "Diese Umgebung überspringt den Login-Bildschirm. Setze AUTH_ENABLED=true und konfiguriere deine Auth-Geheimnisse, um eine Anmeldung zu erzwingen.",
      enterApplication: "{app} öffnen",
      login: "Anmeldung",
      welcomeTo: "Willkommen bei {app}",
      signInAsRole: "Melde dich als {role} mit dem in den Umgebungsvariablen konfigurierten Operator-Konto an.",
      username: "Benutzername",
      password: "Passwort",
      enterUsername: "Benutzernamen eingeben",
      enterPassword: "Passwort eingeben",
      signingIn: "Melde an...",
      signIn: "Anmelden",
      platformFallback: "der Plattform",
      applicationFallback: "Anwendung",
    },
    settingsPage: {
      title: "Einstellungen",
      subtitleFallback: "Konto- und Systemeinstellungen an einem Ort",
      appSubtitle: "{app}-Einstellungen",
      active: "aktiv",
      debugCurrentSettings: "Debug: Aktuelle Einstellungen",
      lastUpdated: "Zuletzt aktualisiert: {value}",
    },
    settingsPanel: {
      language: "Sprache",
      toolSettings: "Werkzeug-Einstellungen",
      loadingTools: "Werkzeuge werden geladen...",
      ragConfiguration: "RAG-Konfiguration",
      loadingDefaults: "Standardwerte werden geladen...",
      knowledgeCollection: "Wissenssammlung (Systemstandard: {value})",
      topKResults: "Top-K-Ergebnisse (Systemstandard: {default}): {value}",
      similarityThreshold: "Ähnlichkeitsschwelle",
      modelSelection: "Modellauswahl",
      systemDefault: "Systemstandard: {value}",
      loadingModels: "Modelle werden geladen...",
      noModelsAvailable: "Keine Modelle verfügbar - läuft Ollama?",
      saveSettings: "Einstellungen speichern",
      settingsSaved: "Einstellungen gespeichert",
      failedToSaveSettings: "Einstellungen konnten nicht gespeichert werden",
      reingestSectionTitle: "Wissens-Neuindizierung",
      reingestDescription: "Startet eine vollstandige Neuindizierung aller aktuell verfugbaren Dokumente aus der Ingestionsquelle.",
      reingestAllDocuments: "Alle Dokumente neu ingestieren",
      reingesting: "Neuindizierung lauft...",
      confirmReingestAll: "Dies leert die konfigurierte Collection und indiziert sie neu. Fortfahren?",
      reingestSuccess: "Neuindizierung abgeschlossen: {processed} Dateien, {chunks} Chunks",
      reingestFailed: "Neuindizierung der Dokumente fehlgeschlagen",
    },
    chat: {
      noMessagesYet: "Noch keine Nachrichten",
      startConversationHint: "Starte eine Unterhaltung oder wähle unten eine Vorlage",
      aiAgent: "KI-Agent",
      aiThinking: "KI denkt nach",
      excludeFromNextMessage: "Von der nächsten Nachricht ausschließen",
      includeInNextMessage: "In nächste Nachricht einbeziehen",
      deleteAttachment: "Anhang löschen",
      uploadingAttachment: "Anhang wird hochgeladen...",
      message: "Nachricht",
      typeYourMessage: "Deine Nachricht eingeben...",
      sendMessage: "Nachricht senden",
      send: "Senden",
      attachmentCountOne: "{count} Anhang ausgewählt",
      attachmentCountOther: "{count} Anhänge ausgewählt",
      toggleWorkspaceSidebar: "Workspace-Seitenleiste umschalten",
      workspace: "Workspace",
      newConversation: "Neue Unterhaltung",
      copyMessage: "Nachricht kopieren",
      regenerateResponse: "Antwort neu generieren",
      responseMetrics: "Antwortmetrik",
      responseTime: "Antwortzeit",
      inputTokens: "Eingabe-Token",
      outputTokens: "Ausgabe-Token",
      finishReason: "Beendigungsgrund",
      model: "Modell",
      temperature: "Temperatur",
      toolsInvoked: "Verwendete Tools",
      feedbackSaved: "Feedback gespeichert",
      feedbackRemoved: "Feedback entfernt",
      workspaceDocumentSaved: "Workspace-Dokument gespeichert",
      messageFailed: "Nachricht fehlgeschlagen",
      failedToSendMessage: "Senden der Nachricht fehlgeschlagen",
      couldNotCreateConversationForAttachment: "Unterhaltung für Anhang konnte nicht erstellt werden",
      attachmentUploadFailed: "Hochladen des Anhangs fehlgeschlagen",
      attachmentUploaded: "Anhang hochgeladen",
      attachmentDeleteFailed: "Löschen des Anhangs fehlgeschlagen",
      attachmentRemoved: "Anhang entfernt",
      templates: {
        explainConcept: "Konzept erklären",
        helpCode: "Beim Coden helfen",
        summarizeText: "Text zusammenfassen",
        translate: "Übersetzen",
        brainstormIdeas: "Ideen sammeln",
        debugIssue: "Problem debuggen",
        explainPrompt: "Erkläre, wie ",
        codePrompt: "Schreibe eine Funktion, die ",
        summarizePrompt: "Fasse den folgenden Text zusammen:\n\n",
        translatePrompt: "Übersetze Folgendes ins Englische:\n\n",
        brainstormPrompt: "Gib mir kreative Ideen für ",
        debugPrompt: "Hilf mir beim Debuggen dieses Problems:\n\n",
      },
    },
    workspace: {
      loadFailed: "Laden fehlgeschlagen",
      saveFailed: "Speichern fehlgeschlagen",
      deleteFailed: "Löschen fehlgeschlagen",
      downloaded: "Heruntergeladen",
      reattachFailed: "Erneutes Anhängen fehlgeschlagen",
      attachedToChat: "An Chat angehängt",
      commandAdded: "Befehl zum Editor hinzugefügt",
      uploadFailed: "Hochladen fehlgeschlagen",
      documentUploaded: "Dokument hochgeladen",
      toggleSidebar: "Workspace-Dokumentseitenleiste umschalten",
      closeSidebar: "Workspace-Seitenleiste schließen",
      uploadDocument: "Dokument hochladen",
      uploading: "Wird hochgeladen...",
      myDocuments: "Meine Dokumente ({count})",
      edit: "Bearbeiten",
      reference: "Referenz",
      download: "Herunterladen",
      delete: "Löschen",
      editor: "Editor",
      closeEditor: "Editor schließen",
      resizeEditor: "Zum Ändern der Größe ziehen",
      editDocumentPlaceholder: "Dokumentinhalt bearbeiten...",
      saveChanges: "Änderungen speichern",
      attachToChat: "An Chat anhängen",
      noDocuments: "Noch keine Dokumente",
      uploadToGetStarted: "Lade Dokumente hoch oder erstelle neue, um zu starten",
      noActiveConversation: "Keine aktive Unterhaltung.",
      loadDocumentFailed: "Dokument konnte nicht geladen werden",
      updateDocumentFailed: "Dokument konnte nicht aktualisiert werden",
      deleteDocumentFailed: "Dokument konnte nicht gelöscht werden",
      saveChangesFailed: "Änderungen konnten nicht gespeichert werden",
      uploaded: "Hochgeladen",
      commandReferenceTemplate:
        "Beziehe dich in deiner Analyse auf das Workspace-Dokument \"{filename}\" (ID: {id}).",
    },
    exportDialog: {
      exportFailedNoData: "Export fehlgeschlagen - keine Daten zurückgegeben.",
      exportFailedTryAgain: "Export fehlgeschlagen. Bitte versuche es erneut.",
      exportedSuccessfully: "Erfolgreich exportiert",
      exportFailed: "Export fehlgeschlagen",
      title: "Unterhaltung exportieren",
      closeDialog: "Dialog schließen",
      format: "Format",
      markdown: "Markdown",
      markdownDescription: "Menschenlesbare .md-Datei",
      json: "JSON",
      jsonDescription: "Strukturierte Daten mit Metadaten",
      exporting: "Exportiere...",
    },
    metrics: {
      title: "{app}-Metrik",
      subtitle: "Systemleistung und Analysen für {role}",
      guardrailHealthyTitle: "Profil-Guardrail in Ordnung",
      guardrailWarningTitle: "Profil-Guardrail Warnung",
      guardrailHealthyHint: "Keine Profilabweichungen erkannt",
      guardrailWarningHint: "Profilabweichungen erkannt",
      sectionsLabel: "Metrikbereiche",
      realtimeTab: "Echtzeit-Metrik",
      trendsTab: "Analyse-Trends",
      realtimeTitle: "Echtzeit-Metrik",
      trendsTitle: "Analyse-Trends",
      loadingMetrics: "Metrik werden geladen...",
      loadingTrends: "Trends werden geladen...",
      savePreferences: "Präferenzen werden gespeichert...",
      preferencesSaved: "Präferenzen gespeichert",
      preferencesFailed: "Speichern fehlgeschlagen",
      dateRange: "Datumsbereich",
      invalidRange: "Ungültiger Bereich",
      last7Days: "Letzte 7 Tage",
      last30Days: "Letzte 30 Tage",
      last90Days: "Letzte 90 Tage",
      autoRefresh60s: "Automatisch aktualisieren (60s)",
      lastUpdated: "Zuletzt aktualisiert",
      export: "Exportieren",
      totalRequests: "Gesamtanfragen",
      totalTokens: "Gesamt-Token",
      avgLatency: "Durchschn. Latenz",
      activeSessions: "Aktive Sitzungen",
      attachmentsInjected: "Eingefügte Anhänge",
      requestsWithoutAttachments: "Anfragen ohne Anhänge",
      attachmentRetryTriggers: "Anhang-Retry-Auslöser",
      attachmentRetrySuccess: "Anhang-Retry-Erfolg",
      profileIdMismatches: "Profil-ID-Abweichungen",
      memoryOperations: "Speicheroperationen",
      llmCallsByProvider: "LLM-Aufrufe nach Anbieter",
      providerModelStatus: "Anbieter/Modell/Status",
      count: "Anzahl",
      attachmentContextMetrics: "Anhang-Kontextmetrik",
      attachmentRetryGuardrail: "Anhang-Retry-Guardrail",
      usageTrends: "Nutzungstrends",
      dailyRequestsAndUsers: "Tägliche Anfragen und eindeutige Nutzer",
      requests: "Anfragen",
      users: "Nutzer",
      tokenConsumption: "Token-Verbrauch",
      totalAndAverageTokens: "Gesamt- und Durchschnittstoken pro Tag",
      total: "Gesamt",
      average: "Durchschnitt",
      latencyAnalysis: "Latenzanalyse",
      minAverageMaxDuration: "Minimale, durchschnittliche und maximale Anfragedauer",
      min: "Min",
      max: "Max",
      memoryTrends: "Speichertrends",
      dailyMemoryOpsQualityAndErrorRate: "Tägliche Ladevorgänge, Updates, Fehler, Qualität und Fehlerrate",
      noMemoryTrendData: "Keine Speichertrenddaten verfügbar",
      summary: "Zusammenfassung",
      keyMetricsForPeriod: "Wichtige Kennzahlen für den ausgewählten Zeitraum",
      uniqueUsers: "Eindeutige Nutzer",
      memoryLoads: "Speicher-Ladevorgänge",
      memoryUpdates: "Speicher-Updates",
      memoryQueries: "Speicher-Abfragen",
      memoryErrorRate: "Speicher-Fehlerrate",
      totalOps: "Gesamtvorgänge",
      avgQuality: "Durchschn. Qualität",
      coOccurrenceNodes: "Ko-Vorkommen Knoten",
      coOccurrenceEdges: "Ko-Vorkommen Kanten",
      rankingLatency: "Ranking-Latenz",
      operation: "Vorgang",
      success: "Erfolg",
      error: "Fehler",
      noUsageData: "Keine Nutzungsdaten verfügbar",
      noTokenData: "Keine Token-Daten verfügbar",
      noLatencyData: "Keine Latenzdaten verfügbar",
      noSummaryData: "Keine Zusammenfassungsdaten verfügbar",
      selectAtLeastOneMetric: "Mindestens eine Kennzahl auswählen",
      errorLoadingTrends: "Fehler beim Laden der Trends",
      days: "Tage",
      na: "k.A.",
      retrievalFeedback: "Abruf-Feedback-Schleife",
      retrievalFeedbackSubtitle: "Wie oft abgerufene Erinnerungen anschließend vom Nutzer bewertet werden",
      retrievalSignalsTotal: "Abrufsignale",
      retrievedWithRating: "Mit Bewertung abgerufen",
      retrievedWithoutRating: "Ohne Bewertung abgerufen",
      priorRatingCoverage: "Vorherige Bewertungsabdeckung",
      ratedAfterRetrieval: "Nach Abruf bewertet",
      feedbackCoverage: "Feedback-Abdeckung",
      ratingBuckets: "Bewertungskategorien beim Abruf",
      bucketNone: "Nicht bewertet",
      bucketLow: "Niedrig (1–2)",
      bucketMid: "Mittel (3)",
      bucketHigh: "Hoch (4–5)",
      noRetrievalData: "Noch keine Abrufqualitätsdaten vorhanden",
    },
    memories: {
      title: "Speicher",
      subtitle: "Erinnerungen, die der Agent über dich gelernt hat",
      refresh: "Aktualisieren",
      total: "Gesamt",
      recent7d: "Letzte 7 Tage",
      rated: "Bewertet",
      unrated: "Unbewertet",
      avgImportance: "Durchschn. Wichtigkeit",
      coverage: "{count}% Abdeckung",
      filterPlaceholder: "Erinnerungen filtern...",
      memory: "Speicher",
      importance: "Wichtigkeit",
      rating: "Bewertung",
      saved: "Gespeichert",
      loading: "Wird geladen...",
      noMemoriesMatchFilter: "Keine Erinnerungen entsprechen deinem Filter.",
      noMemoriesYet: "Noch keine Erinnerungen.",
      related: "verwandt",
      deleteMemory: "Speicher löschen",
      confirmYes: "Ja",
      confirmNo: "Nein",
      pageOfTotal: "Seite {page} von {pages} ({total} gesamt)",
      previous: "Zurück",
      next: "Weiter",
      rateStars: "{count} Sterne bewerten",
    },
    charts: {
      loading: "Wird geladen...",
      noData: "Keine Daten verfügbar",
      selectAtLeastOneSeries: "Mindestens eine Reihe auswählen",
      dailyUsageTrends: "Tägliche Nutzungstrends",
      dailyTokenConsumption: "Täglicher Token-Verbrauch",
      requestLatencyAnalysisMs: "Anfrage-Latenzanalyse (ms)",
      requestsByStatus: "Anfragen nach Status",
      noRequestData: "Keine Anfragedaten verfügbar",
      tokenUsage: "Token-Nutzung",
      noTokenUsageData: "Keine Token-Nutzungsdaten verfügbar",
      tokenUsageByModel: "Token-Nutzung nach Modell",
      total: "Gesamt",
      tokens: "Token",
      loads: "Ladevorgänge",
      updates: "Updates",
      errors: "Fehler",
      errorRatePercent: "Fehlerrate %",
      qualityPercent: "Qualität %",
      loadingMemoryTrends: "Speichertrends werden geladen...",
      noMemoryTrendData: "Keine Speichertrenddaten verfügbar",
      maxLatency: "Max-Latenz",
      avgLatency: "Durchschn.-Latenz",
      minLatency: "Min-Latenz",
    },
  },
};
