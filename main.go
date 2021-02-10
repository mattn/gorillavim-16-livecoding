package main

import (
	"bufio"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/goml/gobrain"
	"github.com/goml/gobrain/persist"
)

func loadData() ([][]float64, []string, error) {
	f, err := os.Open("iris.csv")
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// skip header
	scanner.Scan()

	var resultf [][]float64
	var results []string
	for scanner.Scan() {
		var f1, f2, f3, f4 float64
		var s string
		n, err := fmt.Sscanf(scanner.Text(), "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, &s)
		if n != 5 || err != nil {
			return nil, nil, errors.New("cannot load data")
		}

		resultf = append(resultf, []float64{f1, f2, f3, f4})
		results = append(results, strings.Trim(s, `"`))
	}
	return resultf, results, nil
}

func shuffle(x [][]float64, y []string) {
	for i := len(x) - 1; i >= 0; i-- {
		j := rand.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}
}

func main() {
	rand.Seed(time.Now().Unix())

	X, Y, err := loadData()
	if err != nil {
		log.Fatal(err)
	}

	shuffle(X, Y)

	n := 100
	xtrain, ytrain, xtest, ytest := X[:n], Y[:n], X[n:], Y[n:]

	patterns := [][][]float64{}

	m := map[string][]float64{
		"Setosa":     {1, 0, 0},
		"Versicolor": {0, 1, 0},
		"Virginica":  {0, 0, 1},
	}

	for i, x := range xtrain {
		patterns = append(patterns, [][]float64{
			x, m[ytrain[i]],
		})
	}
	ff := &gobrain.FeedForward{}
	ff.Init(4, 3, 3)

	err = persist.Load("model.json", &ff)
	if err != nil {
		ff.Train(patterns, 100000, 0.6, 0.04, true)
		persist.Save("model.json", &ff)
	}

	var a int
	for i, x := range xtest {
		result := ff.Update(x)
		var mf float64
		var mj int
		for j, v := range result {
			if mf < v {
				mf = v
				mj = j
			}
		}
		want := ytest[i]
		got := []string{"Setosa", "Versicolor", "Virginica"}[mj]
		if want == got {
			a++
		}
	}
	fmt.Printf("%.02f%%\n", float64(a)/float64(len(xtest))*100)
}
