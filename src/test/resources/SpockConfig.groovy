
def VERSION = "0.20.1" // This is changed by gradle.build automatically
def PROJECT = "Neureka"

spockReports {
    set 'com.athaydes.spockframework.report.IReportCreator':'com.athaydes.spockframework.report.template.TemplateReportCreator'

    set 'com.athaydes.spockframework.report.outputDir':'docs/spock/reports' // Output directory (where the spock reports will be created) - relative to working directory
    set 'com.athaydes.spockframework.report.aggregatedJsonReportDir':'docs/spock/aggregated'
    set 'com.athaydes.spockframework.report.internal.HtmlReportCreator.excludeToc':false
    set 'com.athaydes.spockframework.report.hideEmptyBlocks':false // If set to true, hides blocks which do not have any description
    set 'com.athaydes.spockframework.report.projectName':PROJECT
    set 'com.athaydes.spockframework.report.projectVersion':VERSION // Neureka version!

    // Show the source code for each block
    set 'com.athaydes.spockframework.report.showCodeBlocks':true

    // Set the root location of the Spock test source code (only used if showCodeBlocks is 'true')
    set 'com.athaydes.spockframework.report.testSourceRoots':'src/test/groovy'

    // Set properties specific to the TemplateReportCreator
    set 'com.athaydes.spockframework.report.template.TemplateReportCreator.specTemplateFile':'/templates/spec-template.json'
    set 'com.athaydes.spockframework.report.template.TemplateReportCreator.reportFileExtension':'json'
    set 'com.athaydes.spockframework.report.template.TemplateReportCreator.summaryTemplateFile':'/templates/summary-template.json'
    set 'com.athaydes.spockframework.report.template.TemplateReportCreator.summaryFileName':'summary.json'
    set 'com.athaydes.spockframework.report.template.TemplateReportCreator.enabled':true
}