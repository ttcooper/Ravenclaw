#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    #titlePanel("Old Faithful Geyser Data"),
    titlePanel("Machine Learning Techniques for Predicting Wind Power generation from weather data for Germany"),

    # Sidebar with a slider input for number of bins 
    
    sidebarLayout(
        sidebarPanel(
            selectInput(
                inputId  = "check",
                label = "Select the results you want to view",
                choices = c("Results Summary", "Linear Regression Onshore", "Linear Regression Offshore", "Random Forest View 1", "Random Forest View 2", "Neural Network"),
                selected = "Results Summary"
        )),
        
        # Show a plot of the generated distribution
        mainPanel(
            plotOutput("image")
        )
    )
)


# Define server logic required to draw a histogram
server <- function(input, output) {
    output$image <- renderImage({
        if (input$check == "Random Forest View 1") {
            return(list(
                src = "images/RF_results_v1.png",
                contentType = "png",
                alt = "Random Forest View 1",
                height = 300, 
                width = 560
            ))
        } else if (input$check == "Random Forest View 2") {
             return(list(
                src = "images/RF_results_v2.png",
                contentType = "png",
                alt = "Random Forest View 2",
                height = 400, 
                width = 600
                ))

        } else if (input$check == "Neural Network") {
            return(list(
                src = "images/NN_results.png",
                contentType = "png",
                alt = "Neural Network",
                height = 400, 
                width = 560
            ))
            
        } else if (input$check == "Linear Regression Onshore") {
            return(list(
                src = "images/LR_onshore_results.png",
                contentType = "png",
                alt = "Linear Regression Onshore",
                height = 400, 
                width = 600
            ))
            
        } else if (input$check == "Linear Regression Offshore") {
            return(list(
                src = "images/LR_offshore_results.png",
                contentType = "png",
                alt = "Linear Regression Offshore",
                height = 400, 
                width = 600
            ))
            
        } else if (input$check == "Results Summary") {
            return(list(
                src = "images/results_compare.png",
                contentType = "png",
                alt = "Results Summary",
                height = 110, 
                width = 420
            ))
        } 
    }, deleteFile = FALSE)
}

# Run the application 
shinyApp(ui = ui, server = server)

#rsconnect::deployApp('/Users/tingcooper/PycharmProjects/solar_prediction/ShinyApp')
