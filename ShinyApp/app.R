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
                label = "Select the Machine Learning Technique",
                choices = c("Linear Regression", "Random Forest", "Neural Network"),
                selected = "Random Forest"
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
        if (input$check == "Random Forest") {
            return(list(
                src = "images/RF_results.png",
                contentType = "png",
                alt = "Random Forest",
                height = 400, 
                width = 560
            ))
        } else if (input$check == "Neural Network") {
            return(list(
                src = "images/NN_results.png",
                contentType = "png",
                alt = "Neural Network",
                height = 400, 
                width = 560
            ))
        } else if (input$check == "Linear Regression") {
            return(list(
                src = "images/LR_results.png",
                contentType = "png",
                alt = "Linear Regression",
                height = 300, 
                width = 560
            ))
        } 
    }, deleteFile = FALSE)
}

# Run the application 
shinyApp(ui = ui, server = server)

#rsconnect::deployApp('/Users/tingcooper/PycharmProjects/solar_prediction/ShinyApp')
