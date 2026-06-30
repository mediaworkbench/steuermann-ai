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
    confirm: string;
    delete: string;
  };
  confirmDialog: {
    typeToConfirmLabel: string;
    typeToConfirmPlaceholder: string;
    resetCheckboxLabel: string;
  };
  header: {
    metrics: string;
    chats: string;
    memory: string;
    settings: string;
    admin: string;
    ragExplorer: string;
    openNavigation: string;
    activeSession: string;
    exportConversation: string;
    exportChat: string;
    signOut: string;
    signingOut: string;
    justNow: string;
    minutesAgo: string;
    hoursAgo: string;
    daysAgo: string;
    messageCountOne: string;
    messageCountOther: string;
  };
  ragExplorer: {
    title: string;
    subtitle: string;
    searchPlaceholder: string;
    search: string;
    searching: string;
    collection: string;
    topK: string;
    resultsSummary: string;
    noResults: string;
    emptyState: string;
    cutoffDivider: string;
    aboveCutoff: string;
    belowCutoff: string;
    chunk: string;
    copy: string;
    copied: string;
    showMore: string;
    showLess: string;
  };
  sidebar: {
    closeNavigation: string;
    startNewChat: string;
    newChat: string;
    chatHistory: string;
    pinned: string;
    recentChats: string;
    noConversations: string;
    seeAll: string;
    administration: string;
    settingsForUser: string;
    moreOptions: string;
    rename: string;
    pin: string;
    unpin: string;
    delete: string;
    deleteConversationConfirm: string;
  };
  chats: {
    title: string;
    subtitle: string;
    searchPlaceholder: string;
    searching: string;
    loading: string;
    colTitle: string;
    colMessages: string;
    colUpdated: string;
    noChatsYet: string;
    noMatch: string;
    selectRow: string;
    selectAllVisible: string;
    clearSelection: string;
    selectedCount: string;
    pinSelected: string;
    unpinSelected: string;
    deleteSelected: string;
    deleteSelectedConfirm: string;
    openChat: string;
    refresh: string;
    pageOfTotal: string;
    previous: string;
    next: string;
    viewEvidence: string;
    closeEvidence: string;
    evidenceLoading: string;
    evidenceError: string;
    evidenceEmpty: string;
    evidenceLatestHint: string;
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
  adminPage: {
    title: string;
    subtitle: string;
    profileLabel: string;
    frameworkVersionLabel: string;
    llmSection: string;
    ragSection: string;
    modelSection: string;
    dangerZoneSection: string;
    dangerZoneDescription: string;
    resetConversationsLabel: string;
    resetConversationsDescription: string;
    resetWorkspaceLabel: string;
    resetWorkspaceDescription: string;
    resetMemoriesLabel: string;
    resetMemoriesDescription: string;
    resetAnalyticsLabel: string;
    resetAnalyticsDescription: string;
    resetLlmProbesLabel: string;
    resetLlmProbesDescription: string;
    resetSelectedButton: string;
    resetNoneSelected: string;
    resetConfirmMessage: string;
    accessDenied: string;
    heartbeatSection: string;
    heartbeatDescription: string;
    heartbeatRateLabel: string;
    heartbeatRateHelp: string;
    heartbeatDisabledNote: string;
    heartbeatDefaultNote: string;
    heartbeatLastRunLabel: string;
    heartbeatNever: string;
    heartbeatSave: string;
    heartbeatSaved: string;
    heartbeatSaveFailed: string;
    heartbeatLoadFailed: string;
    heartbeatRateTitle: string;
    heartbeatTasksTitle: string;
    heartbeatTasksHelp: string;
    heartbeatScopeGlobal: string;
    heartbeatScopePerUser: string;
    heartbeatTaskDisabled: string;
    heartbeatCooldownLabel: string;
    heartbeatNoTasks: string;
    heartbeatLogTitle: string;
    heartbeatLogHelp: string;
    heartbeatColTask: string;
    heartbeatColUser: string;
    heartbeatColStatus: string;
    heartbeatColDuration: string;
    heartbeatColWhen: string;
    heartbeatFilterAllTasks: string;
    heartbeatFilterAllUsers: string;
    heartbeatFilterAllStatuses: string;
    heartbeatGlobalUser: string;
    heartbeatNoRuns: string;
    heartbeatRefresh: string;
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
    noToolsAvailable: string;
    ragConfiguration: string;
    ragEnabled: string;
    loadingDefaults: string;
    knowledgeCollection: string;
    topKResults: string;
    similarityThreshold: string;
    modelSelection: string;
    systemDefault: string;
    loadingModels: string;
    noModelsAvailable: string;
    noRoleModelsAvailable: string;
    roleModelLabel: string;
    roleProviderLocked: string;
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
    resetSectionTitle: string;
    resetDescription: string;
    resetAllDatabases: string;
    resetting: string;
    confirmReset: string;
    resetSuccess: string;
    resetFailed: string;
    capabilitiesTitle: string;
    capabilitiesSubtitle: string;
    capabilitiesTtl: string;
    capabilitiesLoading: string;
    capabilitiesLoadFailed: string;
    capabilitiesEmpty: string;
    legendTitle: string;
    legendNative: string;
    legendStructured: string;
    legendReact: string;
    copyDiagnostics: string;
    copyDiagnosticsSuccess: string;
    copyDiagnosticsFailed: string;
    capabilityDetails: string;
    showDetails: string;
    hideDetails: string;
    detailConfiguredMode: string;
    detailApiBase: string;
    detailError: string;
    detailBindTools: string;
    detailVision: string;
    detailReasoning: string;
    detailMismatch: string;
    detailMetadata: string;
    capabilityModel: string;
    capabilityRole: string;
    capabilityDesired: string;
    capabilityEffective: string;
    capabilityProbeStatus: string;
    capabilityReason: string;
    capabilityProbedAt: string;
    interfaceSection: string;
    chatSection: string;
    themeLabel: string;
    themeLight: string;
    themeDark: string;
    themeAuto: string;
    showMetricsLabel: string;
    showMetricsDescription: string;
    soundEnabled: string;
    soundEnabledDescription: string;
    myDataSection: string;
    myDataDescription: string;
    myDataResetConversationsLabel: string;
    myDataResetConversationsDescription: string;
    myDataResetWorkspaceLabel: string;
    myDataResetWorkspaceDescription: string;
    myDataResetMemoriesLabel: string;
    myDataResetMemoriesDescription: string;
    myDataResetButton: string;
    myDataResetting: string;
    myDataResetSuccess: string;
    myDataResetFailed: string;
    myDataResetNoneSelected: string;
    myDataResetConfirmMessage: string;
  };
  chat: {
    noMessagesYet: string;
    startConversationHint: string;
    aiAgent: string;
    aiThinking: string;
    insertReference: string;
    deleteAttachment: string;
    uploadingAttachment: string;
    message: string;
    typeYourMessage: string;
    sendMessage: string;
    stopGenerating: string;
    queueMessage: string;
    queued: string;
    queuedHint: string;
    queuedSlotFull: string;
    sendQueuedNow: string;
    cancelQueued: string;
    editQueued: string;
    send: string;
    addFile: string;
    addImage: string;
    addAttachment: string;
    toggleWorkspaceSidebar: string;
    toolsMenuTitle: string;
    noToolsEnabled: string;
    workspace: string;
    newConversation: string;
    copyMessage: string;
    regenerateResponse: string;
    responseMetrics: string;
    responseTime: string;
    inputTokens: string;
    outputTokens: string;
    tokensPerSecond: string;
    finishReason: string;
    model: string;
    temperature: string;
    toolsInvoked: string;
    memoriesUsed: string;
    memoriesUsedDetail: string;
    feedbackSaved: string;
    feedbackRemoved: string;
    workspaceDocumentSaved: string;
    updatingDocument: string;
    messageFailed: string;
    failedToSendMessage: string;
    couldNotCreateConversationForAttachment: string;
    attachmentUploadFailed: string;
    attachmentUploaded: string;
    attachmentDeleteFailed: string;
    attachmentRemoved: string;
    knowledgeBaseOn: string;
    knowledgeBaseOff: string;
    memoryOn: string;
    memoryOff: string;
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
  providerHealth: {
    offlineTitle: string;
    degradedTitle: string;
    retry: string;
    checking: string;
    composerHint: string;
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
    splitViewTitle: string;
    closeSplitView: string;
    resizeSplitView: string;
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
    attach: string;
    attachSuccess: string;
    attachFailed: string;
    nukeAll: string;
    nukeMessage: string;
    nukeSuccess: string;
    nukeFailed: string;
    uploadAriaLabel: string;
    thumbnailClickHint: string;
    documentActions: string;
    tabDocuments: string;
    tabKnowledge: string;
    tabMemory: string;
    tabOutputs: string;
    knowledgeEmpty: string;
    memoryEmpty: string;
    outputsEmpty: string;
    searchDocuments: string;
    clearSearch: string;
    noResults: string;
    noResultsHint: string;
    loadingDocuments: string;
    loadingVersions: string;
    documentsLoadError: string;
    retry: string;
    versionHistoryTitle: string;
    noSavedVersions: string;
    preview: string;
    hidePreview: string;
    restore: string;
    restoredToVersion: string;
    restoreFailed: string;
    rename: string;
    renamed: string;
    renameFailed: string;
    history: string;
    versionHistoryTooltip: string;
    loadVersionHistoryFailed: string;
    loadVersionPreviewFailed: string;
    closeLightbox: string;
    evidenceSummary: string;
    evidenceDocs: string;
    evidenceMemory: string;
    evidenceTools: string;
    evidenceSources: string;
    evidenceAttachments: string;
    knowledgeRetrieved: string;
    knowledgeNoResults: string;
    sourcesHeading: string;
    sourceCitationLabel: string;
    documentsInContext: string;
    attachmentsInContext: string;
    mapGenerated: string;
    openFullMap: string;
    mapDistanceBadge: string;
    weatherFeelsLike: string;
    weatherHumidity: string;
    weatherWind: string;
    weatherForecastTitle: string;
    weatherWarmerBy: string;
    weatherSameTemp: string;
    weatherClear: string;
    weatherPartlyCloudy: string;
    weatherOvercast: string;
    weatherFog: string;
    weatherDrizzle: string;
    weatherRain: string;
    weatherSnow: string;
    weatherShowers: string;
    weatherThunderstorm: string;
    weatherUnknownCondition: string;
    metaScore: string;
    metaRated: string;
    tabInspector: string;
    inspectorEmpty: string;
    inspectorRunning: string;
    inspectorNodes: string;
    inspectorNotRun: string;
    inspectorPostResponse: string;
    inspectorPostResponseRunning: string;
    inspectorWaiting: string;
    inspectorStatusSuccess: string;
    inspectorStatusError: string;
    answerEvidence: string;
    toolArgs: string;
    toolOutput: string;
    toolError: string;
    viewToolResults: string;
    viewingEarlierAnswer: string;
    jumpToLatest: string;
    unsavedChanges: string;
    saveConflict: string;
    writebackWhileDirty: string;
    previewTruncated: string;
    versionSourceUser: string;
    versionSourceAssistant: string;
    versionSourceRestored: string;
    versionCurrent: string;
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
    coverageHealthy: string;
    coverageLow: string;
    coverageVeryLow: string;
    messageQuality: string;
    messageQualitySubtitle: string;
    thumbsUp: string;
    thumbsDown: string;
    netScore: string;
    feedbackRate: string;
    totalFeedback: string;
    noMessageQualityData: string;
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
    primary: string;
    deleteMemory: string;
    confirmYes: string;
    confirmNo: string;
    pageOfTotal: string;
    previous: string;
    next: string;
    rateStars: string;
    ratingHelp: string;
    ratingAriaLabel: string;
    ratingStatusSaving: string;
    ratingStatusSaved: string;
    ratingStatusRetry: string;
    emptyHint: string;
    clearAll: string;
    clearAllConfirmMessage: string;
    clearAllSuccess: string;
    clearAllFailed: string;
  };
  dreaming: {
    userTitle: string;
    userSubtitle: string;
    adminTitle: string;
    adminDescription: string;
    refresh: string;
    loadError: string;
    conflictsTitle: string;
    conflictsDescription: string;
    conflictsEmpty: string;
    conflictEstablished: string;
    conflictNew: string;
    keepOld: string;
    acceptNew: string;
    dependsLabel: string;
    proceduralTitle: string;
    proceduralDescription: string;
    proceduralEmpty: string;
    approve: string;
    reject: string;
    tierLabel: string;
    tier3Locked: string;
    statusProposed: string;
    statusActive: string;
    statusObserving: string;
    statusRejected: string;
    undoTitle: string;
    undoDescription: string;
    undoEmpty: string;
    undoButton: string;
    expiresIn: string;
    actionDelete: string;
    actionLowerConfidence: string;
    actionPromote: string;
    actionPropose: string;
    cyclesRun: string;
    vectorCount: string;
    avgCost: string;
    totalTokens: string;
    pendingResolutions: string;
    openConflicts: string;
    proposedRules: string;
    deletions: string;
    promotions: string;
    actionSuccess: string;
    actionFailed: string;
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
  toolGroups: {
    text: string;
    vision: string;
    auxiliary: string;
  };
  roleTools: {
    title: string;
    description: string;
    save: string;
    saved: string;
    saveFailed: string;
    loadFailed: string;
    loading: string;
    roleUser: string;
    roleResearcher: string;
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
      confirm: "Confirm",
      delete: "Delete",
    },
    confirmDialog: {
      typeToConfirmLabel: "Type {word} to confirm",
      typeToConfirmPlaceholder: "Type here...",
      resetCheckboxLabel: "I understand this will permanently delete all data",
    },
    header: {
      metrics: "Metrics",
      chats: "Chats",
      memory: "Memory",
      settings: "Settings",
      admin: "Admin",
      ragExplorer: "RAG Explorer",
      openNavigation: "Open navigation",
      activeSession: "Active session",
      exportConversation: "Export conversation",
      exportChat: "Export Chat",
      signOut: "Sign out",
      signingOut: "Signing out...",
      justNow: "just now",
      minutesAgo: "{count}m ago",
      hoursAgo: "{count}h ago",
      daysAgo: "{count}d ago",
      messageCountOne: "{count} msg",
      messageCountOther: "{count} msgs",
    },
    ragExplorer: {
      title: "RAG Knowledge Explorer",
      subtitle:
        "Search the knowledge base by keyword and review the matching documents for evaluation.",
      searchPlaceholder: "Search the knowledge base…",
      search: "Search",
      searching: "Searching…",
      collection: "Collection",
      topK: "Results",
      resultsSummary: "{count} result(s) in \"{collection}\"",
      noResults: "No matching documents.",
      emptyState: "Enter a keyword and search to explore the knowledge base.",
      cutoffDivider: "Production cutoff · {threshold}",
      aboveCutoff: "above cutoff",
      belowCutoff: "below cutoff",
      chunk: "chunk {index}/{count}",
      copy: "Copy chunk text",
      copied: "Copied",
      showMore: "Show more",
      showLess: "Show less",
    },
    sidebar: {
      closeNavigation: "Close navigation",
      startNewChat: "Start a new chat",
      newChat: "New Chat",
      chatHistory: "Chat history",
      pinned: "Pinned",
      recentChats: "Recent Chats",
      noConversations: "No conversations yet",
      seeAll: "See all",
      administration: "Administration",
      settingsForUser: "Open settings for {name}",
      moreOptions: "More options",
      rename: "Rename",
      pin: "Pin",
      unpin: "Unpin",
      delete: "Delete",
      deleteConversationConfirm: "Delete this conversation?",
    },
    chats: {
      title: "Chats",
      subtitle: "Browse, search and manage all your conversations",
      searchPlaceholder: "Search message content...",
      searching: "Searching...",
      loading: "Loading chats...",
      colTitle: "Title",
      colMessages: "Messages",
      colUpdated: "Updated",
      noChatsYet: "No conversations yet",
      noMatch: "No chats match your search",
      selectRow: "Select conversation",
      selectAllVisible: "Select all",
      clearSelection: "Clear",
      selectedCount: "{count} selected",
      pinSelected: "Pin",
      unpinSelected: "Unpin",
      deleteSelected: "Delete",
      deleteSelectedConfirm: "Delete {count} conversation(s)? This cannot be undone.",
      openChat: "Open chat",
      refresh: "Refresh",
      pageOfTotal: "Page {page} of {pages} · {total} chats",
      previous: "Previous",
      next: "Next",
      viewEvidence: "View answer evidence",
      closeEvidence: "Close evidence panel",
      evidenceLoading: "Loading answer evidence…",
      evidenceError: "Couldn't load this conversation.",
      evidenceEmpty: "No assistant answer in this conversation yet.",
      evidenceLatestHint: "Evidence from this conversation's latest answer.",
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
      signInAsRole: "Sign in with your credentials provided by your Administrator.",
      username: "Username",
      password: "Password",
      enterUsername: "Enter your username",
      enterPassword: "Enter your password",
      signingIn: "Signing in...",
      signIn: "Sign in",
      platformFallback: "the platform",
      applicationFallback: "application",
    },
    adminPage: {
      title: "Setup & Administration",
      subtitle: "Diagnostics, operational tuning, and system maintenance",
      profileLabel: "Profile: {profile}",
      frameworkVersionLabel: "Framework version: {version}",
      llmSection: "Model Tool-Calling Capabilities",
      ragSection: "Knowledge Base Configuration",
      modelSection: "System Model Selection",
      dangerZoneSection: "Danger Zone",
      dangerZoneDescription: "Select the data categories to purge. All users are affected. This cannot be undone.",
      resetConversationsLabel: "Conversations & Messages",
      resetConversationsDescription: "All conversations, messages, and file attachments.",
      resetWorkspaceLabel: "Workspace & Documents",
      resetWorkspaceDescription: "All workspace documents, versions, and files on disk.",
      resetMemoriesLabel: "Memories & Knowledge Base",
      resetMemoriesDescription: "All Qdrant vector collections (long-term memories and RAG knowledge base).",
      resetAnalyticsLabel: "Analytics",
      resetAnalyticsDescription: "All analytics events and daily aggregates.",
      resetLlmProbesLabel: "LLM Capability Probes",
      resetLlmProbesDescription: "Cached model capability probe results (re-probed automatically on next request).",
      resetSelectedButton: "Reset Selected",
      resetNoneSelected: "Select at least one category to reset.",
      resetConfirmMessage: "Permanently deletes the selected data categories for all users. The schema is preserved. This cannot be undone.",
      accessDenied: "You do not have permission to view this page.",
      heartbeatSection: "Heartbeat",
      heartbeatDescription: "A virtual cron: the agent wakes on this schedule and runs its background tasks. Changes apply within ~30 seconds.",
      heartbeatRateLabel: "Beat rate (minutes)",
      heartbeatRateHelp: "How often the heartbeat wakes (1–1440 minutes).",
      heartbeatDisabledNote: "The heartbeat is disabled for the active profile.",
      heartbeatDefaultNote: "Using the profile default ({minutes} min).",
      heartbeatLastRunLabel: "Last beat",
      heartbeatNever: "No beats recorded yet",
      heartbeatSave: "Save",
      heartbeatSaved: "Heartbeat rate updated",
      heartbeatSaveFailed: "Failed to update heartbeat rate",
      heartbeatLoadFailed: "Failed to load heartbeat settings",
      heartbeatRateTitle: "Beat rate",
      heartbeatTasksTitle: "Configured tasks",
      heartbeatTasksHelp: "Tasks run on each beat. Per-user tasks fan out once per active user.",
      heartbeatScopeGlobal: "Global",
      heartbeatScopePerUser: "Per-user",
      heartbeatTaskDisabled: "Disabled",
      heartbeatCooldownLabel: "Cooldown {seconds}s",
      heartbeatNoTasks: "No tasks configured.",
      heartbeatLogTitle: "Run log",
      heartbeatLogHelp: "Most recent beats, newest first. Expand an error row for details.",
      heartbeatColTask: "Task",
      heartbeatColUser: "User",
      heartbeatColStatus: "Status",
      heartbeatColDuration: "Duration",
      heartbeatColWhen: "When",
      heartbeatFilterAllTasks: "All tasks",
      heartbeatFilterAllUsers: "All users",
      heartbeatFilterAllStatuses: "All statuses",
      heartbeatGlobalUser: "—",
      heartbeatNoRuns: "No beats recorded yet.",
      heartbeatRefresh: "Refresh",
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
      noToolsAvailable: "No tools available.",
      ragConfiguration: "RAG Configuration",
      ragEnabled: "Enable Knowledge Base",
      loadingDefaults: "Loading defaults...",
      knowledgeCollection: "Knowledge Collection (System default: {value})",
      topKResults: "Top-K Results (System default: {default}): {value}",
      similarityThreshold: "Similarity Threshold",
      modelSelection: "Model Selection",
      systemDefault: "System default: {value}",
      loadingModels: "Loading models...",
      noModelsAvailable: "No models available - is Ollama running?",
      noRoleModelsAvailable: "No role model configuration available",
      roleModelLabel: "{role} model",
      roleProviderLocked: "Provider locked by profile: {provider}",
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
      resetSectionTitle: "Reset All Databases",
      resetDescription: "Permanently deletes all conversations, memories, workspace documents, analytics, and Qdrant collections. The schema is preserved. This cannot be undone.",
      resetAllDatabases: "Reset All Databases",
      resetting: "Resetting...",
      confirmReset: "Type RESET to confirm permanent deletion of all data",
      resetSuccess: "Selected data reset successfully",
      resetFailed: "Reset failed",
      capabilitiesTitle: "Model Tool-Calling Capabilities",
      capabilitiesSubtitle: "Desired vs effective mode based on probe results. ",
      capabilitiesTtl: "Probe TTL: {value}s",
      capabilitiesLoading: "Loading model capability status...",
      capabilitiesLoadFailed: "Failed to load model capability status",
      capabilitiesEmpty: "No model capability results available yet",
      legendTitle: "Legend:",
      legendNative: "native",
      legendStructured: "structured",
      legendReact: "react",
      copyDiagnostics: "Copy diagnostics",
      copyDiagnosticsSuccess: "Diagnostics copied",
      copyDiagnosticsFailed: "Failed to copy diagnostics",
      capabilityDetails: "Details",
      showDetails: "Show",
      hideDetails: "Hide",
      detailConfiguredMode: "Configured mode",
      detailApiBase: "API base",
      detailError: "Probe error",
      detailBindTools: "Supports bind_tools",
      detailVision: "Supports vision",
      detailReasoning: "Supports reasoning",
      detailMismatch: "Capability mismatch",
      detailMetadata: "Raw metadata",
      capabilityModel: "Model",
      capabilityRole: "Role",
      capabilityDesired: "Desired",
      capabilityEffective: "Effective",
      capabilityProbeStatus: "Probe Status",
      capabilityReason: "Reason",
      capabilityProbedAt: "Probed At",
      interfaceSection: "Interface",
      chatSection: "Chat",
      themeLabel: "Color scheme",
      themeLight: "Light",
      themeDark: "Dark",
      themeAuto: "System default",
      showMetricsLabel: "Show response metrics",
      showMetricsDescription: "Display timing, token count, and model info under each answer",
      soundEnabled: "Enable interface sounds",
      soundEnabledDescription: "Play a sound notification when the assistant replies",
      myDataSection: "My Data",
      myDataDescription: "Permanently delete your personal data. This cannot be undone.",
      myDataResetConversationsLabel: "Conversations & Messages",
      myDataResetConversationsDescription: "All your chat conversations and messages",
      myDataResetWorkspaceLabel: "Workspace & Documents",
      myDataResetWorkspaceDescription: "All your workspace files and documents",
      myDataResetMemoriesLabel: "Memories",
      myDataResetMemoriesDescription: "All memories the assistant has built about you",
      myDataResetButton: "Delete Selected",
      myDataResetting: "Deleting…",
      myDataResetSuccess: "Your selected data was deleted successfully.",
      myDataResetFailed: "Failed to delete your data. Please try again.",
      myDataResetNoneSelected: "Select at least one category to delete.",
      myDataResetConfirmMessage: "This will permanently delete your selected personal data. This action cannot be undone.",
    },
    chat: {
      noMessagesYet: "No messages yet",
      startConversationHint: "Start a conversation or pick a template below",
      aiAgent: "AI Agent",
      aiThinking: "AI is thinking",
      insertReference: "Insert reference into message",
      deleteAttachment: "Delete attachment",
      uploadingAttachment: "Uploading attachment...",
      message: "Message",
      typeYourMessage: "Type your message...",
      sendMessage: "Send message",
      stopGenerating: "Stop generating",
      queueMessage: "Queue message",
      queued: "Queued",
      queuedHint: "Type a follow-up…",
      queuedSlotFull: "Message queued — send or remove it first",
      sendQueuedNow: "Send now",
      cancelQueued: "Remove queued message",
      editQueued: "Edit queued message",
      send: "Send",
      addFile: "Add file",
      addImage: "Add image",
      addAttachment: "Add attachment",
      toggleWorkspaceSidebar: "Toggle workspace sidebar",
      toolsMenuTitle: "Tools",
      noToolsEnabled: "No tools enabled — turn them on in Settings.",
      workspace: "Workspace & Provenance",
      newConversation: "New conversation",
      copyMessage: "Copy message",
      regenerateResponse: "Regenerate this response",
      responseMetrics: "Response metrics",
      responseTime: "Response time",
      inputTokens: "Input tokens",
      outputTokens: "Output tokens",
      tokensPerSecond: "Tokens/sec",
      finishReason: "Finish reason",
      model: "Model",
      temperature: "Temperature",
      toolsInvoked: "Tools invoked",
      memoriesUsed: "memories",
      memoriesUsedDetail: "Memories used",
      feedbackSaved: "Feedback saved",
      feedbackRemoved: "Feedback removed",
      workspaceDocumentSaved: "Workspace document saved",
      updatingDocument: "Updating {name}…",
      messageFailed: "Message failed",
      failedToSendMessage: "Failed to send message",
      couldNotCreateConversationForAttachment: "Could not create conversation for attachment",
      attachmentUploadFailed: "Attachment upload failed",
      attachmentUploaded: "Attachment uploaded",
      attachmentDeleteFailed: "Attachment delete failed",
      attachmentRemoved: "Attachment removed",
      knowledgeBaseOn: "Knowledge Base: On",
      knowledgeBaseOff: "Knowledge Base: Off",
      memoryOn: "Memory: On",
      memoryOff: "Memory: Off",
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
    providerHealth: {
      offlineTitle: "LLM provider offline — chat is unavailable",
      degradedTitle: "Some LLM services are unavailable — responses may be degraded",
      retry: "Retry",
      checking: "Checking…",
      composerHint: "Chat is unavailable while the provider is offline",
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
      splitViewTitle: "Active document",
      closeSplitView: "Close split view",
      resizeSplitView: "Drag to resize the document pane",
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
      commandReferenceTemplate: "\"{filename}\" (id: {id})",
      attach: "Attach",
      attachSuccess: "Attached to conversation",
      attachFailed: "Could not attach file",
      nukeAll: "Delete all",
      nukeMessage: "This permanently deletes all {count} document(s). This cannot be undone.",
      nukeSuccess: "Deleted {count} file(s)",
      nukeFailed: "Could not clear workspace",
      uploadAriaLabel: "Upload document or image",
      thumbnailClickHint: "Click to preview",
      documentActions: "Document actions",
      tabDocuments: "Documents",
      tabKnowledge: "Knowledge",
      tabMemory: "Memory",
      tabOutputs: "Outputs",
      knowledgeEmpty: "Knowledge sources used in answers will appear here.",
      memoryEmpty: "Memories recalled for answers will appear here.",
      outputsEmpty: "Tool and generation outputs will appear here.",
      searchDocuments: "Search documents",
      clearSearch: "Clear search",
      noResults: "No matching documents",
      noResultsHint: "Try a different search term.",
      loadingDocuments: "Loading documents…",
      loadingVersions: "Loading…",
      documentsLoadError: "Couldn't load documents",
      retry: "Retry",
      versionHistoryTitle: "Version History - {name}",
      noSavedVersions: "No saved versions yet.",
      preview: "Preview",
      hidePreview: "Hide",
      restore: "Restore",
      restoredToVersion: "Restored to v{version}",
      restoreFailed: "Restore failed",
      rename: "Rename",
      renamed: "Renamed",
      renameFailed: "Rename failed",
      history: "History",
      versionHistoryTooltip: "Version history",
      loadVersionHistoryFailed: "Failed to load version history",
      loadVersionPreviewFailed: "Failed to load version preview",
      closeLightbox: "Close",
      evidenceSummary: "Answer evidence",
      evidenceDocs: "Documents used",
      evidenceMemory: "Memories recalled",
      evidenceTools: "Tools used",
      evidenceSources: "Sources",
      evidenceAttachments: "Attachments used",
      knowledgeRetrieved: "{count} document(s) retrieved",
      knowledgeNoResults: "Searched · no relevant results",
      sourcesHeading: "Sources",
      sourceCitationLabel: "Citation {n}",
      documentsInContext: "Documents in context",
      attachmentsInContext: "Attachments in context",
      mapGenerated: "Map output",
      openFullMap: "Open full map",
      mapDistanceBadge: "{km} km · {miles} mi",
      weatherFeelsLike: "Feels like {temp}",
      weatherHumidity: "Humidity",
      weatherWind: "Wind",
      weatherForecastTitle: "{days}-day forecast",
      weatherWarmerBy: "{place} is {delta} warmer",
      weatherSameTemp: "Same temperature in both places",
      weatherClear: "Clear sky",
      weatherPartlyCloudy: "Partly cloudy",
      weatherOvercast: "Overcast",
      weatherFog: "Fog",
      weatherDrizzle: "Drizzle",
      weatherRain: "Rain",
      weatherSnow: "Snow",
      weatherShowers: "Showers",
      weatherThunderstorm: "Thunderstorm",
      weatherUnknownCondition: "Unknown",
      metaScore: "score {score}",
      metaRated: "rated {rating}/5",
      tabInspector: "Inspector",
      inspectorEmpty: "Node execution will appear here after the next answer.",
      inspectorRunning: "Running…",
      inspectorNodes: "{count} nodes",
      inspectorNotRun: "Not in this run",
      inspectorPostResponse: "Runs after the response is sent",
      inspectorPostResponseRunning: "Running in the background…",
      inspectorWaiting: "Capturing node execution…",
      inspectorStatusSuccess: "succeeded",
      inspectorStatusError: "failed",
      answerEvidence: "Answer evidence",
      toolArgs: "Arguments",
      toolOutput: "Result",
      toolError: "Error",
      viewToolResults: "View results in Outputs",
      viewingEarlierAnswer: "Viewing an earlier answer",
      jumpToLatest: "Jump to latest",
      unsavedChanges: "Unsaved changes",
      saveConflict: "The document changed elsewhere — the latest version was reloaded",
      writebackWhileDirty: "The assistant saved a new version, but you have unsaved edits — save to see the conflict, or close to load the new version",
      previewTruncated: "(truncated)",
      versionSourceUser: "You",
      versionSourceAssistant: "AI",
      versionSourceRestored: "Restored",
      versionCurrent: "Current",
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
      retrievalFeedbackSubtitle: "How often memories retrieved in chat are subsequently rated on the Memories page",
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
      coverageHealthy: "Healthy",
      coverageLow: "Low coverage",
      coverageVeryLow: "Very low",
      messageQuality: "Message Quality",
      messageQualitySubtitle: "Daily thumbs up/down feedback on assistant responses",
      thumbsUp: "Thumbs Up",
      thumbsDown: "Thumbs Down",
      netScore: "Net Score",
      feedbackRate: "Feedback Rate",
      totalFeedback: "Total Feedback",
      noMessageQualityData: "No message quality data yet",
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
      primary: "primary",
      deleteMemory: "Delete memory",
      confirmYes: "Yes",
      confirmNo: "No",
      pageOfTotal: "Page {page} of {pages} ({total} total)",
      previous: "Previous",
      next: "Next",
      rateStars: "Rate {count} stars",
      ratingHelp: "Rate memories by how useful they are for future answers.",
      ratingAriaLabel: "Rate memory usefulness for future answers",
      ratingStatusSaving: "Saving",
      ratingStatusSaved: "Saved",
      ratingStatusRetry: "Retry",
      emptyHint: "Continue chatting and the agent will learn from your conversations.",
      clearAll: "Clear All Memories",
      clearAllConfirmMessage: "This will permanently delete all your memories. The assistant will start fresh with no knowledge about you.",
      clearAllSuccess: "All memories cleared.",
      clearAllFailed: "Failed to clear memories. Please try again.",
    },
    dreaming: {
      userTitle: "Memory Review",
      userSubtitle: "Resolve contradictions, approve learned preferences, and undo recent memory changes.",
      adminTitle: "Dreaming Metrics",
      adminDescription: "Aggregate, anonymized health of the background memory engine. No user content is shown.",
      refresh: "Refresh",
      loadError: "Could not load. Please try again.",
      conflictsTitle: "Memory Conflicts",
      conflictsDescription: "The engine found new information that may contradict something it remembers. How should it reconcile them?",
      conflictsEmpty: "No conflicts to resolve.",
      conflictEstablished: "Remembered",
      conflictNew: "New observation",
      keepOld: "Keep remembered",
      acceptNew: "Accept new",
      dependsLabel: "It depends",
      proceduralTitle: "Learned Preferences",
      proceduralDescription: "Behavioural preferences the engine learned from how you interact. They only affect responses once you approve them.",
      proceduralEmpty: "No preferences awaiting review.",
      approve: "Approve",
      reject: "Reject",
      tierLabel: "Tier {tier}",
      tier3Locked: "Core-logic and safety rules are never auto-learned.",
      statusProposed: "Awaiting approval",
      statusActive: "Active",
      statusObserving: "Observing",
      statusRejected: "Rejected",
      undoTitle: "Recent Changes",
      undoDescription: "Memory changes the engine made recently. You can undo each one within its window.",
      undoEmpty: "Nothing to undo.",
      undoButton: "Undo",
      expiresIn: "Undo available for {time}",
      actionDelete: "Forgot a memory",
      actionLowerConfidence: "Lowered confidence",
      actionPromote: "Consolidated memories",
      actionPropose: "Proposed a preference",
      cyclesRun: "Engine cycles run",
      vectorCount: "Stored memory vectors",
      avgCost: "Avg tokens / cycle",
      totalTokens: "Total tokens",
      pendingResolutions: "Pending resolutions",
      openConflicts: "Open conflicts",
      proposedRules: "Proposed rules",
      deletions: "Memories forgotten",
      promotions: "Memories consolidated",
      actionSuccess: "Done.",
      actionFailed: "Action failed. Please try again.",
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
    toolGroups: {
      text: "Text tools",
      vision: "Vision tools",
      auxiliary: "Auxiliary tools",
    },
    roleTools: {
      title: "Role tool access",
      description: "Choose which tools each role may use. Users can still turn allowed tools on or off individually. Administrators always have access to every tool.",
      save: "Save",
      saved: "Tool access saved",
      saveFailed: "Failed to save tool access",
      loadFailed: "Failed to load tool access",
      loading: "Loading tools...",
      roleUser: "User",
      roleResearcher: "Researcher",
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
      confirm: "Bestätigen",
      delete: "Löschen",
    },
    confirmDialog: {
      typeToConfirmLabel: "Gib {word} ein um zu bestätigen",
      typeToConfirmPlaceholder: "Hier eingeben...",
      resetCheckboxLabel: "Ich verstehe, dass alle Daten dauerhaft gelöscht werden",
    },
    header: {
      metrics: "Metrik",
      chats: "Chats",
      memory: "Speicher",
      settings: "Einstellungen",
      admin: "Admin",
      ragExplorer: "RAG-Explorer",
      openNavigation: "Navigation öffnen",
      activeSession: "Aktive Sitzung",
      exportConversation: "Unterhaltung exportieren",
      exportChat: "Chat exportieren",
      signOut: "Abmelden",
      signingOut: "Melde ab...",
      justNow: "gerade eben",
      minutesAgo: "vor {count} Min.",
      hoursAgo: "vor {count} Std.",
      daysAgo: "vor {count} T.",
      messageCountOne: "{count} Nachricht",
      messageCountOther: "{count} Nachrichten",
    },
    ragExplorer: {
      title: "RAG-Wissensexplorer",
      subtitle:
        "Durchsuche die Wissensbasis nach Stichwörtern und prüfe die passenden Dokumente zur Evaluierung.",
      searchPlaceholder: "Wissensbasis durchsuchen…",
      search: "Suchen",
      searching: "Suche läuft…",
      collection: "Sammlung",
      topK: "Treffer",
      resultsSummary: "{count} Treffer in \"{collection}\"",
      noResults: "Keine passenden Dokumente.",
      emptyState: "Gib ein Stichwort ein und suche, um die Wissensbasis zu erkunden.",
      cutoffDivider: "Produktions-Schwellenwert · {threshold}",
      aboveCutoff: "über Schwelle",
      belowCutoff: "unter Schwelle",
      chunk: "Chunk {index}/{count}",
      copy: "Chunk-Text kopieren",
      copied: "Kopiert",
      showMore: "Mehr anzeigen",
      showLess: "Weniger anzeigen",
    },
    sidebar: {
      closeNavigation: "Navigation schließen",
      startNewChat: "Neuen Chat starten",
      newChat: "Neuer Chat",
      chatHistory: "Chat-Verlauf",
      pinned: "Angeheftet",
      recentChats: "Letzte Chats",
      noConversations: "Noch keine Unterhaltungen",
      seeAll: "Alle anzeigen",
      administration: "Verwaltung",
      settingsForUser: "Einstellungen für {name} öffnen",
      moreOptions: "Weitere Optionen",
      rename: "Umbenennen",
      pin: "Anheften",
      unpin: "Lösen",
      delete: "Löschen",
      deleteConversationConfirm: "Diese Unterhaltung löschen?",
    },
    chats: {
      title: "Chats",
      subtitle: "Alle Unterhaltungen durchsuchen und verwalten",
      searchPlaceholder: "Nachrichteninhalt durchsuchen...",
      searching: "Suche läuft...",
      loading: "Chats werden geladen...",
      colTitle: "Titel",
      colMessages: "Nachrichten",
      colUpdated: "Aktualisiert",
      noChatsYet: "Noch keine Unterhaltungen",
      noMatch: "Keine Chats entsprechen der Suche",
      selectRow: "Unterhaltung auswählen",
      selectAllVisible: "Alle auswählen",
      clearSelection: "Aufheben",
      selectedCount: "{count} ausgewählt",
      pinSelected: "Anheften",
      unpinSelected: "Lösen",
      deleteSelected: "Löschen",
      deleteSelectedConfirm: "{count} Unterhaltung(en) löschen? Dies kann nicht rückgängig gemacht werden.",
      openChat: "Chat öffnen",
      refresh: "Aktualisieren",
      pageOfTotal: "Seite {page} von {pages} · {total} Chats",
      previous: "Zurück",
      next: "Weiter",
      viewEvidence: "Antwort-Belege ansehen",
      closeEvidence: "Belege-Panel schließen",
      evidenceLoading: "Antwort-Belege werden geladen…",
      evidenceError: "Diese Unterhaltung konnte nicht geladen werden.",
      evidenceEmpty: "Noch keine Assistenten-Antwort in dieser Unterhaltung.",
      evidenceLatestHint: "Belege aus der letzten Antwort dieser Unterhaltung.",
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
      signInAsRole: "Melde dich mit deinen von deinem Administrator bereitgestellten Anmeldedaten an.",
      username: "Benutzername",
      password: "Passwort",
      enterUsername: "Benutzernamen eingeben",
      enterPassword: "Passwort eingeben",
      signingIn: "Melde an...",
      signIn: "Anmelden",
      platformFallback: "der Plattform",
      applicationFallback: "Anwendung",
    },
    adminPage: {
      title: "Setup & Administration",
      subtitle: "Diagnose, Betriebstuning und Systemwartung",
      profileLabel: "Profil: {profile}",
      frameworkVersionLabel: "Framework-Version: {version}",
      llmSection: "Modell Tool-Calling-Fähigkeiten",
      ragSection: "Wissensdatenbank-Konfiguration",
      modelSection: "Systemmodell-Auswahl",
      dangerZoneSection: "Gefahrenbereich",
      dangerZoneDescription: "Wähle die zu löschenden Datenkategorien aus. Alle Benutzer sind betroffen. Dies kann nicht rückgängig gemacht werden.",
      resetConversationsLabel: "Konversationen & Nachrichten",
      resetConversationsDescription: "Alle Konversationen, Nachrichten und Dateianhänge.",
      resetWorkspaceLabel: "Workspace & Dokumente",
      resetWorkspaceDescription: "Alle Workspace-Dokumente, Versionen und Dateien auf dem Datenträger.",
      resetMemoriesLabel: "Erinnerungen & Wissensdatenbank",
      resetMemoriesDescription: "Alle Qdrant-Vektorkollektionen (Langzeiterinnerungen und RAG-Wissensdatenbank).",
      resetAnalyticsLabel: "Analysen",
      resetAnalyticsDescription: "Alle Analyseereignisse und tägliche Aggregate.",
      resetLlmProbesLabel: "LLM-Fähigkeitsprüfungen",
      resetLlmProbesDescription: "Zwischengespeicherte Modell-Fähigkeitsprüfergebnisse (werden beim nächsten Aufruf automatisch neu abgerufen).",
      resetSelectedButton: "Auswahl zurücksetzen",
      resetNoneSelected: "Bitte mindestens eine Kategorie auswählen.",
      resetConfirmMessage: "Löscht dauerhaft die ausgewählten Datenkategorien aller Benutzer. Das Schema bleibt erhalten. Diese Aktion kann nicht rückgängig gemacht werden.",
      accessDenied: "Sie haben keine Berechtigung, diese Seite anzuzeigen.",
      heartbeatSection: "Heartbeat",
      heartbeatDescription: "Ein virtueller Cron: Der Agent wacht in diesem Takt auf und führt seine Hintergrundaufgaben aus. Änderungen werden innerhalb von ca. 30 Sekunden übernommen.",
      heartbeatRateLabel: "Taktrate (Minuten)",
      heartbeatRateHelp: "Wie oft der Heartbeat aufwacht (1–1440 Minuten).",
      heartbeatDisabledNote: "Der Heartbeat ist für das aktive Profil deaktiviert.",
      heartbeatDefaultNote: "Profilstandard wird verwendet ({minutes} Min.).",
      heartbeatLastRunLabel: "Letzter Takt",
      heartbeatNever: "Noch keine Takte aufgezeichnet",
      heartbeatSave: "Speichern",
      heartbeatSaved: "Heartbeat-Takt aktualisiert",
      heartbeatSaveFailed: "Heartbeat-Takt konnte nicht aktualisiert werden",
      heartbeatLoadFailed: "Heartbeat-Einstellungen konnten nicht geladen werden",
      heartbeatRateTitle: "Taktrate",
      heartbeatTasksTitle: "Konfigurierte Aufgaben",
      heartbeatTasksHelp: "Aufgaben laufen bei jedem Takt. Pro-Benutzer-Aufgaben fächern pro aktivem Benutzer auf.",
      heartbeatScopeGlobal: "Global",
      heartbeatScopePerUser: "Pro Benutzer",
      heartbeatTaskDisabled: "Deaktiviert",
      heartbeatCooldownLabel: "Abklingzeit {seconds}s",
      heartbeatNoTasks: "Keine Aufgaben konfiguriert.",
      heartbeatLogTitle: "Ausführungsprotokoll",
      heartbeatLogHelp: "Neueste Takte zuerst. Fehlerzeile zum Aufklappen der Details anklicken.",
      heartbeatColTask: "Aufgabe",
      heartbeatColUser: "Benutzer",
      heartbeatColStatus: "Status",
      heartbeatColDuration: "Dauer",
      heartbeatColWhen: "Wann",
      heartbeatFilterAllTasks: "Alle Aufgaben",
      heartbeatFilterAllUsers: "Alle Benutzer",
      heartbeatFilterAllStatuses: "Alle Status",
      heartbeatGlobalUser: "—",
      heartbeatNoRuns: "Noch keine Takte aufgezeichnet.",
      heartbeatRefresh: "Aktualisieren",
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
      noToolsAvailable: "Keine Werkzeuge verfügbar.",
      ragConfiguration: "RAG-Konfiguration",
      ragEnabled: "Wissensdatenbank aktivieren",
      loadingDefaults: "Standardwerte werden geladen...",
      knowledgeCollection: "Wissenssammlung (Systemstandard: {value})",
      topKResults: "Top-K-Ergebnisse (Systemstandard: {default}): {value}",
      similarityThreshold: "Ähnlichkeitsschwelle",
      modelSelection: "Modellauswahl",
      systemDefault: "Systemstandard: {value}",
      loadingModels: "Modelle werden geladen...",
      noModelsAvailable: "Keine Modelle verfügbar - läuft Ollama?",
      noRoleModelsAvailable: "Keine Rollen-Modellkonfiguration verfugbar",
      roleModelLabel: "{role}-Modell",
      roleProviderLocked: "Provider durch Profil festgelegt: {provider}",
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
      resetSectionTitle: "Alle Datenbanken zurücksetzen",
      resetDescription: "Löscht dauerhaft alle Konversationen, Erinnerungen, Workspace-Dokumente, Analysen und Qdrant-Collections. Das Schema bleibt erhalten. Diese Aktion kann nicht rückgängig gemacht werden.",
      resetAllDatabases: "Alle Datenbanken zurücksetzen",
      resetting: "Wird zurückgesetzt...",
      confirmReset: "Gib RESET ein, um die dauerhafte Löschung aller Daten zu bestätigen",
      resetSuccess: "Ausgewählte Daten erfolgreich zurückgesetzt",
      resetFailed: "Zurücksetzen fehlgeschlagen",
      capabilitiesTitle: "Modell Tool-Calling-Fahigkeiten",
      capabilitiesSubtitle: "Gewunschter vs effektiver Modus basierend auf Probe-Ergebnissen. ",
      capabilitiesTtl: "Probe-TTL: {value}s",
      capabilitiesLoading: "Lade Modell-Fahigkeitsstatus...",
      capabilitiesLoadFailed: "Modell-Fahigkeitsstatus konnte nicht geladen werden",
      capabilitiesEmpty: "Noch keine Modell-Fahigkeitsergebnisse verfugbar",
      legendTitle: "Legende:",
      legendNative: "native",
      legendStructured: "structured",
      legendReact: "react",
      copyDiagnostics: "Diagnose kopieren",
      copyDiagnosticsSuccess: "Diagnose kopiert",
      copyDiagnosticsFailed: "Diagnose konnte nicht kopiert werden",
      capabilityDetails: "Details",
      showDetails: "Anzeigen",
      hideDetails: "Verbergen",
      detailConfiguredMode: "Konfigurierter Modus",
      detailApiBase: "API-Basis",
      detailError: "Probe-Fehler",
      detailBindTools: "Unterstutzt bind_tools",
      detailVision: "Unterstützt Vision",
      detailReasoning: "Unterstützt Reasoning",
      detailMismatch: "Fahigkeitskonflikt",
      detailMetadata: "Roh-Metadaten",
      capabilityModel: "Modell",
      capabilityRole: "Rolle",
      capabilityDesired: "Gewunscht",
      capabilityEffective: "Effektiv",
      capabilityProbeStatus: "Probe-Status",
      capabilityReason: "Grund",
      capabilityProbedAt: "Gepruft am",
      interfaceSection: "Oberfläche",
      chatSection: "Chat",
      themeLabel: "Farbschema",
      themeLight: "Hell",
      themeDark: "Dunkel",
      themeAuto: "Systemstandard",
      showMetricsLabel: "Antwortmetriken anzeigen",
      showMetricsDescription: "Antwortzeit, Token-Anzahl und Modellinformationen unter jeder Antwort anzeigen",
      soundEnabled: "Schnittstellentöne aktivieren",
      soundEnabledDescription: "Einen Ton abspielen, wenn der Assistent antwortet",
      myDataSection: "Meine Daten",
      myDataDescription: "Persönliche Daten dauerhaft löschen. Dies kann nicht rückgängig gemacht werden.",
      myDataResetConversationsLabel: "Gespräche & Nachrichten",
      myDataResetConversationsDescription: "Alle deine Chatgespräche und Nachrichten",
      myDataResetWorkspaceLabel: "Arbeitsbereich & Dokumente",
      myDataResetWorkspaceDescription: "Alle deine Arbeitsbereich-Dateien und Dokumente",
      myDataResetMemoriesLabel: "Erinnerungen",
      myDataResetMemoriesDescription: "Alle Erinnerungen, die der Assistent über dich gesammelt hat",
      myDataResetButton: "Ausgewähltes löschen",
      myDataResetting: "Wird gelöscht…",
      myDataResetSuccess: "Deine ausgewählten Daten wurden erfolgreich gelöscht.",
      myDataResetFailed: "Deine Daten konnten nicht gelöscht werden. Bitte erneut versuchen.",
      myDataResetNoneSelected: "Wähle mindestens eine Kategorie aus.",
      myDataResetConfirmMessage: "Dies löscht dauerhaft deine ausgewählten persönlichen Daten. Diese Aktion kann nicht rückgängig gemacht werden.",
    },
    chat: {
      noMessagesYet: "Noch keine Nachrichten",
      startConversationHint: "Starte eine Unterhaltung oder wähle unten eine Vorlage",
      aiAgent: "KI-Agent",
      aiThinking: "KI denkt nach",
      insertReference: "Referenz in Nachricht einfügen",
      deleteAttachment: "Anhang löschen",
      uploadingAttachment: "Anhang wird hochgeladen...",
      message: "Nachricht",
      typeYourMessage: "Deine Nachricht eingeben...",
      sendMessage: "Nachricht senden",
      stopGenerating: "Generierung stoppen",
      queueMessage: "Nachricht einreihen",
      queued: "In Warteschlange",
      queuedHint: "Folgenachricht eingeben…",
      queuedSlotFull: "Nachricht in Warteschlange — zuerst senden oder entfernen",
      sendQueuedNow: "Jetzt senden",
      cancelQueued: "Wartende Nachricht entfernen",
      editQueued: "Wartende Nachricht bearbeiten",
      send: "Senden",
      addFile: "Datei hinzufügen",
      addImage: "Bild hinzufügen",
      addAttachment: "Anhang hinzufügen",
      toggleWorkspaceSidebar: "Workspace-Seitenleiste umschalten",
      toolsMenuTitle: "Werkzeuge",
      noToolsEnabled: "Keine Werkzeuge aktiviert — in den Einstellungen aktivieren.",
      workspace: "Workspace & Provenance",
      newConversation: "Neue Unterhaltung",
      copyMessage: "Nachricht kopieren",
      regenerateResponse: "Antwort neu generieren",
      responseMetrics: "Antwortmetrik",
      responseTime: "Antwortzeit",
      inputTokens: "Eingabe-Token",
      outputTokens: "Ausgabe-Token",
      tokensPerSecond: "Token/Sek.",
      finishReason: "Beendigungsgrund",
      model: "Modell",
      temperature: "Temperatur",
      toolsInvoked: "Verwendete Tools",
      memoriesUsed: "Erinnerungen",
      memoriesUsedDetail: "Verwendete Erinnerungen",
      feedbackSaved: "Feedback gespeichert",
      feedbackRemoved: "Feedback entfernt",
      workspaceDocumentSaved: "Workspace-Dokument gespeichert",
      updatingDocument: "Aktualisiere {name}…",
      messageFailed: "Nachricht fehlgeschlagen",
      failedToSendMessage: "Senden der Nachricht fehlgeschlagen",
      couldNotCreateConversationForAttachment: "Unterhaltung für Anhang konnte nicht erstellt werden",
      attachmentUploadFailed: "Hochladen des Anhangs fehlgeschlagen",
      attachmentUploaded: "Anhang hochgeladen",
      attachmentDeleteFailed: "Löschen des Anhangs fehlgeschlagen",
      attachmentRemoved: "Anhang entfernt",
      knowledgeBaseOn: "Wissensdatenbank: An",
      knowledgeBaseOff: "Wissensdatenbank: Aus",
      memoryOn: "Gedächtnis: An",
      memoryOff: "Gedächtnis: Aus",
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
    providerHealth: {
      offlineTitle: "LLM-Anbieter offline — Chat nicht verfügbar",
      degradedTitle: "Einige LLM-Dienste sind nicht verfügbar — Antworten könnten eingeschränkt sein",
      retry: "Erneut versuchen",
      checking: "Wird geprüft…",
      composerHint: "Chat ist nicht verfügbar, solange der Anbieter offline ist",
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
      splitViewTitle: "Aktives Dokument",
      closeSplitView: "Geteilte Ansicht schließen",
      resizeSplitView: "Zum Ändern der Breite ziehen",
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
      commandReferenceTemplate: "\"{filename}\" (ID: {id})",
      attach: "Anhängen",
      attachSuccess: "Zur Konversation hinzugefügt",
      attachFailed: "Datei konnte nicht angehängt werden",
      nukeAll: "Alle löschen",
      nukeMessage: "Dies löscht dauerhaft alle {count} Dokument(e). Dies kann nicht rückgängig gemacht werden.",
      nukeSuccess: "{count} Datei(en) gelöscht",
      nukeFailed: "Workspace konnte nicht geleert werden",
      uploadAriaLabel: "Dokument oder Bild hochladen",
      thumbnailClickHint: "Klicken zum Vergrößern",
      documentActions: "Dokumentaktionen",
      tabDocuments: "Dokumente",
      tabKnowledge: "Wissen",
      tabMemory: "Erinnerungen",
      tabOutputs: "Ausgaben",
      knowledgeEmpty: "Hier erscheinen die in Antworten genutzten Wissensquellen.",
      memoryEmpty: "Hier erscheinen die für Antworten abgerufenen Erinnerungen.",
      outputsEmpty: "Hier erscheinen Tool- und Generierungsausgaben.",
      searchDocuments: "Dokumente suchen",
      clearSearch: "Suche löschen",
      noResults: "Keine passenden Dokumente",
      noResultsHint: "Versuche einen anderen Suchbegriff.",
      loadingDocuments: "Dokumente werden geladen…",
      loadingVersions: "Lädt…",
      documentsLoadError: "Dokumente konnten nicht geladen werden",
      retry: "Erneut versuchen",
      versionHistoryTitle: "Versionsverlauf - {name}",
      noSavedVersions: "Noch keine gespeicherten Versionen.",
      preview: "Vorschau",
      hidePreview: "Verbergen",
      restore: "Wiederherstellen",
      restoredToVersion: "Auf v{version} wiederhergestellt",
      restoreFailed: "Wiederherstellen fehlgeschlagen",
      rename: "Umbenennen",
      renamed: "Umbenannt",
      renameFailed: "Umbenennen fehlgeschlagen",
      history: "Verlauf",
      versionHistoryTooltip: "Versionsverlauf",
      loadVersionHistoryFailed: "Versionsverlauf konnte nicht geladen werden",
      loadVersionPreviewFailed: "Versionsvorschau konnte nicht geladen werden",
      closeLightbox: "Schließen",
      evidenceSummary: "Antwort-Belege",
      evidenceDocs: "Verwendete Dokumente",
      evidenceMemory: "Abgerufene Erinnerungen",
      evidenceTools: "Verwendete Tools",
      evidenceSources: "Quellen",
      evidenceAttachments: "Verwendete Anhänge",
      knowledgeRetrieved: "{count} Dokument(e) abgerufen",
      knowledgeNoResults: "Durchsucht · keine relevanten Treffer",
      sourcesHeading: "Quellen",
      sourceCitationLabel: "Zitat {n}",
      documentsInContext: "Dokumente im Kontext",
      attachmentsInContext: "Anhänge im Kontext",
      mapGenerated: "Kartenausgabe",
      openFullMap: "Karte vollständig öffnen",
      mapDistanceBadge: "{km} km · {miles} mi",
      weatherFeelsLike: "Gefühlt {temp}",
      weatherHumidity: "Luftfeuchtigkeit",
      weatherWind: "Wind",
      weatherForecastTitle: "{days}-Tage-Vorhersage",
      weatherWarmerBy: "{place} ist {delta} wärmer",
      weatherSameTemp: "Gleiche Temperatur an beiden Orten",
      weatherClear: "Klarer Himmel",
      weatherPartlyCloudy: "Teilweise bewölkt",
      weatherOvercast: "Bedeckt",
      weatherFog: "Nebel",
      weatherDrizzle: "Nieselregen",
      weatherRain: "Regen",
      weatherSnow: "Schnee",
      weatherShowers: "Schauer",
      weatherThunderstorm: "Gewitter",
      weatherUnknownCondition: "Unbekannt",
      metaScore: "Score {score}",
      metaRated: "Bewertung {rating}/5",
      tabInspector: "Inspektor",
      inspectorEmpty: "Die Knotenausführung erscheint hier nach der nächsten Antwort.",
      inspectorRunning: "Läuft…",
      inspectorNodes: "{count} Knoten",
      inspectorNotRun: "Nicht in diesem Lauf",
      inspectorPostResponse: "Läuft nach dem Senden der Antwort",
      inspectorPostResponseRunning: "Läuft im Hintergrund…",
      inspectorWaiting: "Knotenausführung wird erfasst…",
      inspectorStatusSuccess: "erfolgreich",
      inspectorStatusError: "fehlgeschlagen",
      answerEvidence: "Antwort-Belege",
      toolArgs: "Argumente",
      toolOutput: "Ergebnis",
      toolError: "Fehler",
      viewToolResults: "Ergebnisse in Ausgaben anzeigen",
      viewingEarlierAnswer: "Frühere Antwort wird angezeigt",
      jumpToLatest: "Zur neuesten springen",
      unsavedChanges: "Ungespeicherte Änderungen",
      saveConflict: "Das Dokument wurde an anderer Stelle geändert — die neueste Version wurde neu geladen",
      writebackWhileDirty: "Die KI hat eine neue Version gespeichert, aber du hast ungespeicherte Änderungen — speichere, um den Konflikt zu sehen, oder schließe, um die neue Version zu laden",
      previewTruncated: "(gekürzt)",
      versionSourceUser: "Du",
      versionSourceAssistant: "KI",
      versionSourceRestored: "Wiederhergestellt",
      versionCurrent: "Aktuell",
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
      retrievalFeedbackSubtitle: "Wie oft im Chat abgerufene Erinnerungen anschließend auf der Speicher-Seite bewertet werden",
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
      coverageHealthy: "Gesund",
      coverageLow: "Geringe Abdeckung",
      coverageVeryLow: "Sehr gering",
      messageQuality: "Nachrichtenqualität",
      messageQualitySubtitle: "Tägliches Daumen-hoch/runter-Feedback zu Assistentenantworten",
      thumbsUp: "Daumen hoch",
      thumbsDown: "Daumen runter",
      netScore: "Nettobewertung",
      feedbackRate: "Feedbackrate",
      totalFeedback: "Gesamtfeedback",
      noMessageQualityData: "Noch keine Nachrichtenqualitätsdaten vorhanden",
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
      primary: "primär",
      deleteMemory: "Speicher löschen",
      confirmYes: "Ja",
      confirmNo: "Nein",
      pageOfTotal: "Seite {page} von {pages} ({total} gesamt)",
      previous: "Zurück",
      next: "Weiter",
      rateStars: "{count} Sterne bewerten",
      ratingHelp: "Bewerte Erinnerungen danach, wie hilfreich sie für zukünftige Antworten sind. Nachrichten-Daumen im Chat bleiben separates Qualitätsfeedback.",
      ratingAriaLabel: "Nützlichkeit dieser Erinnerung für zukünftige Antworten bewerten",
      ratingStatusSaving: "Speichert",
      ratingStatusSaved: "Gespeichert",
      ratingStatusRetry: "Erneut versuchen",
      emptyHint: "Fahre fort zu chatten und der Agent wird aus deinen Gesprächen lernen.",
      clearAll: "Alle Erinnerungen löschen",
      clearAllConfirmMessage: "Dies löscht dauerhaft alle deine Erinnerungen. Der Assistent startet ohne jegliches Wissen über dich neu.",
      clearAllSuccess: "Alle Erinnerungen gelöscht.",
      clearAllFailed: "Erinnerungen konnten nicht gelöscht werden. Bitte erneut versuchen.",
    },
    dreaming: {
      userTitle: "Speicher-Überprüfung",
      userSubtitle: "Widersprüche auflösen, gelernte Präferenzen freigeben und kürzliche Speicheränderungen rückgängig machen.",
      adminTitle: "Dreaming-Metriken",
      adminDescription: "Aggregierte, anonymisierte Übersicht des Hintergrund-Speichermoduls. Es werden keine Nutzerinhalte angezeigt.",
      refresh: "Aktualisieren",
      loadError: "Laden fehlgeschlagen. Bitte erneut versuchen.",
      conflictsTitle: "Speicher-Konflikte",
      conflictsDescription: "Das Modul hat neue Informationen gefunden, die einer Erinnerung widersprechen könnten. Wie soll es sie abgleichen?",
      conflictsEmpty: "Keine Konflikte zu lösen.",
      conflictEstablished: "Erinnert",
      conflictNew: "Neue Beobachtung",
      keepOld: "Erinnerung behalten",
      acceptNew: "Neues übernehmen",
      dependsLabel: "Kommt darauf an",
      proceduralTitle: "Gelernte Präferenzen",
      proceduralDescription: "Verhaltenspräferenzen, die das Modul aus deinen Interaktionen gelernt hat. Sie wirken sich erst nach deiner Freigabe aus.",
      proceduralEmpty: "Keine Präferenzen zur Überprüfung.",
      approve: "Freigeben",
      reject: "Ablehnen",
      tierLabel: "Stufe {tier}",
      tier3Locked: "Kernlogik- und Sicherheitsregeln werden nie automatisch gelernt.",
      statusProposed: "Wartet auf Freigabe",
      statusActive: "Aktiv",
      statusObserving: "Wird beobachtet",
      statusRejected: "Abgelehnt",
      undoTitle: "Kürzliche Änderungen",
      undoDescription: "Speicheränderungen des Moduls. Jede kann innerhalb ihres Zeitfensters rückgängig gemacht werden.",
      undoEmpty: "Nichts rückgängig zu machen.",
      undoButton: "Rückgängig",
      expiresIn: "Rückgängig möglich für {time}",
      actionDelete: "Erinnerung vergessen",
      actionLowerConfidence: "Konfidenz gesenkt",
      actionPromote: "Erinnerungen zusammengeführt",
      actionPropose: "Präferenz vorgeschlagen",
      cyclesRun: "Modul-Zyklen ausgeführt",
      vectorCount: "Gespeicherte Speichervektoren",
      avgCost: "Ø Tokens / Zyklus",
      totalTokens: "Tokens gesamt",
      pendingResolutions: "Offene Vorgänge",
      openConflicts: "Offene Konflikte",
      proposedRules: "Vorgeschlagene Regeln",
      deletions: "Vergessene Erinnerungen",
      promotions: "Zusammengeführte Erinnerungen",
      actionSuccess: "Erledigt.",
      actionFailed: "Aktion fehlgeschlagen. Bitte erneut versuchen.",
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
    toolGroups: {
      text: "Text-Werkzeuge",
      vision: "Vision-Werkzeuge",
      auxiliary: "Hilfs-Werkzeuge",
    },
    roleTools: {
      title: "Werkzeugzugriff nach Rolle",
      description: "Legen Sie fest, welche Werkzeuge jede Rolle verwenden darf. Nutzer können erlaubte Werkzeuge weiterhin einzeln ein- oder ausschalten. Administratoren haben immer Zugriff auf alle Werkzeuge.",
      save: "Speichern",
      saved: "Werkzeugzugriff gespeichert",
      saveFailed: "Werkzeugzugriff konnte nicht gespeichert werden",
      loadFailed: "Werkzeugzugriff konnte nicht geladen werden",
      loading: "Werkzeuge werden geladen...",
      roleUser: "Nutzer",
      roleResearcher: "Forscher",
    },
  },
};
