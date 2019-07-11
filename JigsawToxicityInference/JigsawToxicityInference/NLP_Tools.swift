//
//  NLP_Tools.swift
//  JigsawToxicityInference
//
//  Created by Adrian Tineo on 09/07/2019.
//  Copyright © 2019 adriantineo.com. All rights reserved.
//

import Foundation

// Taken from https://martinmitrevski.com/2017/06/30/natural-language-processing-in-ios/

func tf(urlString: String,
        word: String,
        wordCountings: Dictionary<String, Dictionary<String, Int>>,
        totalWordCount: Int) -> Double {
    
    guard let wordCounting = wordCountings[word] else {
        return Double(Int.min)
    }
    
    guard let occurrences = wordCounting[urlString] else {
        return Double(Int.min)
    }
    
    return Double(occurrences) / Double(totalWordCount)
}

func idf(urlString: String,
         word: String,
         wordCountings: Dictionary<String, Dictionary<String, Int>>,
         totalDocs: Int) -> Double {
    
    guard let wordCounting = wordCountings[word] else {
        return 1
    }
    
    var sum = 0
    for (url, count) in wordCounting {
        if url != urlString {
            sum += count
        }
    }
    
    if sum == 0 {
        return 1
    }
    
    let factor = Double(totalDocs) / Double(sum)
    
    return log(factor)
}

func tdIdf(urlString: String,
           word: String,
           wordCountings: Dictionary<String, Dictionary<String, Int>>,
           totalWordCount: Int,
           totalDocs: Int) -> Double {
    
    return tf(urlString: urlString,
              word: word,
              wordCountings: wordCountings,
              totalWordCount: totalWordCount)
            * idf(urlString: urlString,
                  word: word,
                  wordCountings: wordCountings,
                  totalDocs: totalDocs)
}
