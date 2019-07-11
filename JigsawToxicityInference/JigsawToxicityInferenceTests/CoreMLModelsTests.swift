//
//  CoreMLModelsTests.swift
//  JigsawToxicityInferenceTests
//
//  Created by Adrian Tineo on 24.05.19.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import NaturalLanguage
import XCTest
import CoreML
@testable import JigsawToxicityInference

class CoreMLModelsTests: XCTestCase {
    
    func testPerformanceCoreMLModels() {
        let modelNames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        let models = [try! NLModel(mlModel: toxicClassifier().model),
                      try! NLModel(mlModel: severe_toxicClassifier().model),
                      try! NLModel(mlModel: obsceneClassifier().model),
                      try! NLModel(mlModel: threatClassifier().model),
                      try! NLModel(mlModel: insultClassifier().model),
                      try! NLModel(mlModel: identity_hateClassifier().model)]
                      
        let dataURL = Bundle.main.url(forResource: "inputs_for_classification", withExtension: "csv")!
        let data = try! String(contentsOf: dataURL)
        let inputs = data.split(separator: "\n").map { String($0) } [0..<80000]
        
        print("Inferring for \(inputs.count) entries")

        var toxicCount = 0
        var nonToxicCount = 0
        var failedCount = 0
        
        for (idx, model) in models.enumerated() {
            let modelName = modelNames[idx]
            let start = CFAbsoluteTimeGetCurrent()
            for input in inputs {
                let prediction = model.predictedLabel(for: input)
                if let prediction = prediction {
                    if prediction == modelName {
                        toxicCount += 1
                    } else if prediction == "non-" +  modelName {
                        nonToxicCount += 1
                    } else {
                        XCTFail("ERROR: Predicted something we didn't expect: \(prediction)")
                    }
                } else {
                    failedCount += 1
                }
            }
            let end = CFAbsoluteTimeGetCurrent()
            let diff = end - start
            
            print("INFERENCE STATS FOR MODEL \(modelName):")
            print("Number of inputs recognized as \(modelName): \(toxicCount)")
            print("Number of inputs recognized as non-\(modelName): \(nonToxicCount)")
            print("Number of failed predictions: \(failedCount)")
            print("Classification time for \(modelName) model over test set: \(diff)")
            print("Average classification time: \(Double(diff)/Double(inputs.count))")
        }
    }

    func predict(input: String, sklearnModel: MLModel) -> String {
        // TODO:  implement input adaptation
        return ""
    }
    
//    func testPerformanceSKLearnModels() {
//        let modelNames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
//        let models = [try! NLModel(mlModel: toxicSK().model),
//                      try! NLModel(mlModel: severe_toxicSK().model),
//                      try! NLModel(mlModel: obsceneSK().model),
//                      try! NLModel(mlModel: threatSK().model),
//                      try! NLModel(mlModel: insultSK().model),
//                      try! NLModel(mlModel: identity_hateSK().model)]
//
//        let dataURL = Bundle.main.url(forResource: "inputs_for_classification", withExtension: "csv")!
//        let data = try! String(contentsOf: dataURL)
//        let inputs = data.split(separator: "\n").map { String($0) }
//
//        print("Inferring for \(inputs.count) entries")
//        var toxicCount = 0
//        var nonToxicCount = 0
//        var failedCount = 0
//
//        for (idx, model) in models.enumerated() {
//            let modelName = modelNames[idx]
//            let start = CFAbsoluteTimeGetCurrent()
//            for input in inputs {
//                let prediction = predict(input: input, sklearnModel: model)
//                if let prediction = prediction {
//                    if prediction == modelName {
//                        toxicCount += 1
//                    } else if prediction == "non-" +  modelName {
//                        nonToxicCount += 1
//                    } else {
//                        XCTFail("ERROR: Predicted something we didn't expect: \(prediction)")
//                    }
//                } else {
//                    failedCount += 1
//                }
//            }
//            let end = CFAbsoluteTimeGetCurrent()
//            let diff = end - start
//
//            print("INFERENCE STATS FOR MODEL \(modelName):")
//            print("Number of inputs recognized as \(modelName): \(toxicCount)")
//            print("Number of inputs recognized as non-\(modelName): \(nonToxicCount)")
//            print("Number of failed predictions: \(failedCount)")
//            print("Classification time for \(modelName) model over test set: \(diff)")
//            print("Average classification time: \(Double(diff)/Double(inputs.count))")
//        }
//    }
}
