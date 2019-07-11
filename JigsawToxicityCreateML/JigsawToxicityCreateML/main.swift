//
//  main.swift
//  JigsawToxicity
//
//  Created by Adrian Tineo on 21.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//
import Foundation
import CreateML

protocol MLRegressable {
    func predictions(from data: MLDataTable) throws -> MLUntypedColumn
    var targetColumn: String { get }
}
extension MLRegressor: MLRegressable {}
extension MLLinearRegressor: MLRegressable {}

func createLinearModels(trainingData: MLDataTable, targetColumns: [String], featureColumns: [String]) -> [MLLinearRegressor] {
    return targetColumns.compactMap { targetColumn in
        let parameters = MLLinearRegressor.ModelParameters(validationData: nil,
                                                           maxIterations: 20,
                                                           l1Penalty: 0.0,
                                                           l2Penalty: 0.01,
                                                           stepSize: 1.0,
                                                           convergenceThreshold: 0.01,
                                                           featureRescaling: true)
        let regressor =  try? MLLinearRegressor(trainingData: trainingData,
                                                targetColumn: targetColumn,
                                                featureColumns: featureColumns,
                                                parameters: parameters)
        print("Linear regressor for \"\(targetColumn)\":")
        print(regressor?.description ?? "No model created")
        return regressor
    }
}

func createModels(trainingData: MLDataTable, targetColumns: [String], featureColumns: [String]) -> [MLRegressor] {
    return targetColumns.compactMap { targetColumn in
        let regressor =  try? MLRegressor(trainingData: trainingData,
                                          targetColumn: targetColumn,
                                          featureColumns: featureColumns)
        print("Regressor for \"\(targetColumn)\":")
        print(regressor?.description ?? "no model created")
        return regressor
    }
}

func createTextClassifierModels(trainingData: MLDataTable, targetColumns: [String], featureColumns: [String]) -> [MLTextClassifier] {
    return targetColumns.compactMap { targetColumn in
        let parameters = MLTextClassifier.ModelParameters(validationData: nil,
                                                          algorithm: .maxEnt(revision: 1),
                                                          language: .english)
        let classifier = try? MLTextClassifier(trainingData: trainingData,
                                               textColumn: "comment_text",
                                               labelColumn: targetColumn + "_label",
                                               parameters: parameters)
        print("Classifier for \"\(targetColumn)\":")
        print(classifier?.description ?? "no model created")
        return classifier
    }
}

func showSamplePredictions(models: [MLRegressable], entries: MLDataTable)  {
    models.forEach { model in
        var predictionTable = entries
        let predictions = try? model.predictions(from: entries)
        if let predictions = predictions {
            predictionTable.addColumn(predictions, named: "predictions")
            print(predictionTable)
        } else {
            print("ERROR FOR PREDICTIONS FOR \(model.targetColumn)")
        }
    }
}

func showSamplePredictions(model: MLTextClassifier, entries: MLDataColumn<String>, label: String)  {
    if let predictions = try? model.predictions(from: entries) {
        var predictionTable = MLDataTable()
        predictionTable.addColumn(entries, named: "comment_text")
        predictionTable.addColumn(predictions, named: label)
        dump(table: predictionTable, comment: "SAMPLE PREDICTIONS FOR \(label)")
    } else {
        print("ERROR WHEN GETTING PREDICTIONS FOR \(label)")
    }
}

func addLabelColumn(table: MLDataTable, targetColumns: [String]) -> MLDataTable {
    var table = table
    targetColumns.forEach { targetColumn in
        let newColumn = table[targetColumn].map { $0 == 1 ? targetColumn : "non-" + targetColumn}
        table.addColumn(newColumn, named: targetColumn + "_label")
    }
    
    return table
}

func dump(table: MLDataTable, comment: String) {
    print(comment)
    print(table.description)
    print("SIZE: \(table.size)")
    if let error = table.error {
        print("\(comment) DATA ERROR: \(error)")
    }
}

func generateInputTable(trainingSetPath: URL, parsingOptions: MLDataTable.ParsingOptions, targetColumns: [String]) throws -> MLDataTable {
    var trainingTable = try MLDataTable(contentsOf: trainingSetPath, options: parsingOptions)
    trainingTable = addLabelColumn(table: trainingTable, targetColumns: targetColumns)
    
    return trainingTable
}

func generateScoringTable(scoringSetPath: URL, scoringLabelsPath: URL, parsingOptions: MLDataTable.ParsingOptions, targetColumns: [String]) throws -> MLDataTable {
    var scoringTable = try MLDataTable(contentsOf: scoringSetPath, options: parsingOptions)
    //dump(table: scoringTable, comment: "SCORING DATA")
    
    let scoringLabelsTable = try MLDataTable(contentsOf: scoringLabelsPath, options: parsingOptions)
    //dump(table: scoringLabelsTable, comment: "SCORING LABELS")

    // Compose scoring table and scoring labels
    targetColumns.forEach { column in
        scoringTable.addColumn(scoringLabelsTable[column], named: column)
    }
    // Remove values with -1 (those were not used for scoring in the competition and thus we don't have the actual labels)
    let mask = scoringTable[targetColumns.first!].map { $0 != -1 }  // -1 is copied across all labels, so it suffices with choosing "toxic"
    scoringTable = scoringTable[mask]
    //dump(table: scoringTable, comment: "COMPOSED SCORING DATA")
    
    scoringTable = addLabelColumn(table: scoringTable, targetColumns: targetColumns)
    //dump(table: scoringTable, comment: "TRANSFORMED AND COMPOSED SCORING DATA")

    return scoringTable
}

// Parse command line parameters
let referenceColumns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
var targetColumns = referenceColumns
let numParams = CommandLine.argc
var prefixRows: Int? = nil

if numParams < 2 {
    print("No arguments are passed. Using default.")
} else {
    if numParams > 1 {
        let modelLabel = CommandLine.arguments[1]
        if referenceColumns.contains(modelLabel) {
            print("Proceeding with model label \"\(modelLabel)\"")
            targetColumns = [modelLabel]
        } else {
            print("ERROR. Model label \"\(modelLabel)\" not recognized.")
        }
    }
    
    if numParams > 2 {
        if let prefix = Int(CommandLine.arguments[2]) {
            prefixRows = prefix
            print("Reducing times by choosing first \(prefix) elements for training and scoring.")
        }
    }
}

// 1. Import data
let trainingSetPath = URL(fileURLWithPath: "./Data/train_pre.csv")
let scoringSetPath = URL(fileURLWithPath: "./Data/test_pre.csv")
let scoringLabelsPath = URL(fileURLWithPath: "./Data/test_labels.csv")

//let trainingSet = Bundle.main.url(forResource: "train_pre", withExtension: "csv", subdirectory: "Data")!
//let testSet = Bundle.main.url(forResource: "test_pre", withExtension: "csv", subdirectory: "Data")!

var parsingOptions = MLDataTable.ParsingOptions()
//parsingOptions.skipRows = 3
//parsingOptions.containsHeader = true
parsingOptions.delimiter = ","
parsingOptions.lineTerminator = "\n"

let inputDataTable = try generateInputTable(trainingSetPath: trainingSetPath,
                                              parsingOptions: parsingOptions,
                                              targetColumns: targetColumns)
let (trainingTable, testTable) = inputDataTable.randomSplit(by: 0.8)
dump(table: trainingTable, comment: "TRAINING TABLE")
dump(table: testTable, comment: "TEST TABLE")
let scoringTable = try generateScoringTable(scoringSetPath: scoringSetPath,
                                            scoringLabelsPath: scoringLabelsPath,
                                            parsingOptions: parsingOptions,
                                            targetColumns: targetColumns)
dump(table: scoringTable, comment: "SCORING TABLE")

// 2. Create ML Models

let trainingTablePrefix = (prefixRows != nil) ? trainingTable.prefix(prefixRows!) : trainingTable
let models = createTextClassifierModels(trainingData: trainingTablePrefix,
                                        targetColumns: targetColumns,
                                        featureColumns: ["comment_text"])

// 3. Save model early in case we cut the script short
for (idx, model) in models.enumerated() {
    let label = targetColumns[idx]
    let outputPath = "./Models/\(label)Classifier.mlmodel"
    print("SAVING MODEL \(label) TO \(outputPath)")
    let metadata = MLModelMetadata(author: "Adrian Tineo",
                                   shortDescription: "Classify input text as \(label) or non-\(label)",
        license: "GPL",
        version: "0.1",
        additional: nil)
    try model.write(to: URL(fileURLWithPath: outputPath), metadata: metadata)
}

// 4. Show some predictions
for (idx, model) in models.enumerated() {
    showSamplePredictions(model: model, entries: testTable["comment_text"].prefix(10), label: targetColumns[idx])
}

// 5. Evaluate Models
for (idx, model) in models.enumerated() {
    let evaluation = model.evaluation(on: testTable)
    print("EVALUATION METRICS FOR \(targetColumns[idx]):")
    print(evaluation.description)
    if let error = evaluation.error {
        print("EVALUATION ERROR: \(error.localizedDescription)")
    }
}

// 6. Get model scoring
let scoringTablePrefix = (prefixRows != nil) ? scoringTable.prefix(prefixRows!) : scoringTable
for (idx, model) in models.enumerated() {
    let evaluation = model.evaluation(on: scoringTablePrefix)
    print("SCORING METRICS FOR \(targetColumns[idx]):")
    print(evaluation.description)
    if let error = evaluation.error {
        print("SCORING EVALUATION ERROR: \(error.localizedDescription)")
    }
}



