import unittest
import regpy.linregpy as lrp


class TestLinearRegression(unittest.TestCase):
    """ Collection of tests to verify the linregpy module is
    working as intended.
    """
    def testNoRegression(self):
        with self.assertRaises(ValueError):
            lrp.LinearRegression([], [])

    def testSimpleRegressionLine(self):
        linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
        self.assertEqual(linReg.beta, 1.0)
        self.assertEqual(linReg.inty, 0.0)
        
    def testSimpleAxesNotSorted(self):
        linReg = lrp.LinearRegression([3, 1, 2], [3, 1, 2])
        self.assertEqual(linReg.beta, 1.0)
        self.assertEqual(linReg.inty, 0.0)
        
    def testRandomAxesNotSorted(self):
        linReg = lrp.LinearRegression([5, -10, 3, 23], [3, -2, -6, 12])
        self.assertAlmostEqual(linReg.beta, 0.4635911352329263)
        self.assertAlmostEqual(linReg.inty, -0.683853459972863)
        
    def testLargeRegressionNotSorted(self):
        # still b = 1
        linReg = lrp.LinearRegression([3000000, 1000000, 4000000], [3000000, 1000000, 4000000])
        self.assertEqual(linReg.beta, 1.0)
        self.assertEqual(linReg.inty, 0.0)
        
    def testRandomRegression(self):
        linReg = lrp.LinearRegression([31, -46, 12, -9, -15, 11, 10], 
                                      [-8, -62, -9, -63, 37, -94, 78])
        self.assertAlmostEqual(linReg.beta, 0.5745038167938928)
        self.assertAlmostEqual(linReg.inty, -16.79328244274809)
        
    def testRandomRegression2(self):
        linReg = lrp.LinearRegression([15, -15, -64, -53, 64, 34, -75], 
                                      [54, -16, -93, -42, -5, 37, -69])
        self.assertAlmostEqual(linReg.beta, 0.7979846449136277)
        self.assertAlmostEqual(linReg.inty, -8.427063339731284)
        
    def testRandomRegression3(self):
        linReg = lrp.LinearRegression([-90, 72, -16, 38, 23, 12, -97], 
                                      [42, -99, -4, -65, -6, -92, -39])
        self.assertAlmostEqual(linReg.beta, -0.5268219270415693)
        self.assertAlmostEqual(linReg.inty, -41.93652453834443)
    
    def testSimpleRegressionPredict(self):
        linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
        self.assertEqual(linReg.predict(-1000000000), -1000000000.0)
        self.assertEqual(linReg.predict(-1), -1.0)
        self.assertEqual(linReg.predict(0), 0.0)
        self.assertEqual(linReg.predict(1), 1.0)
        self.assertEqual(linReg.predict(1000000000), 1000000000.0)
        
    def testRandomRegressionPredict(self):
        linReg = lrp.LinearRegression([73, -7, -11, 31, 89, 7, 84], 
                                      [-8, -96, -72, -93, 53, 73, 94])
        self.assertAlmostEqual(linReg.predict(-1000000000), -1141061748.3740551)
        self.assertAlmostEqual(linReg.predict(-1), -51.50140622253471)
        self.assertAlmostEqual(linReg.predict(0), -50.360344524521004)
        self.assertAlmostEqual(linReg.predict(1), -49.219282826507296)
        self.assertAlmostEqual(linReg.predict(1000000000), 1141061647.6533663)
        
    def testRandomRegressionPredict2(self):
        linReg = lrp.LinearRegression([85, -41, 86, 67, 67, -7, -61], 
                                      [58, 89, -60, 5, 74, 74, 80])
        self.assertAlmostEqual(linReg.predict(-1000000000), 549359198.6907732)
        self.assertAlmostEqual(linReg.predict(-1), 61.64570070452423)
        self.assertAlmostEqual(linReg.predict(0), 61.0963415669298)
        self.assertAlmostEqual(linReg.predict(1), 60.54698242933537)
        self.assertAlmostEqual(linReg.predict(1000000000), -549359076.49809)
        
    def testRandomRegressionPredict3(self):
        linReg = lrp.LinearRegression([91, 73, -15, -23, -75, 9, 12], 
                                      [-38, 88, -76, -57, 59, -67, -90])
        self.assertAlmostEqual(linReg.predict(-1000000000), -60118576.29430577)
        self.assertAlmostEqual(linReg.predict(-1), -26.53562363366932)
        self.assertAlmostEqual(linReg.predict(0), -26.47550508385052)
        self.assertAlmostEqual(linReg.predict(1), -26.41538653403172)
        self.assertAlmostEqual(linReg.predict(1000000000), 60118523.343295604)
        
    def testSimpleDomainRange(self):
        linReg = lrp.LinearRegression([0, 1, 3], [2, 3, 4])
        self.assertEqual(linReg.domain, (0.0, 3.0))
        self.assertEqual(linReg.range, (2.0, 4.0))
        
    def testNegativeDomainRange(self):
        linReg = lrp.LinearRegression([-1, -2, -3], [-1, -2, -3])
        self.assertEqual(linReg.domain, (-3.0, -1.0))
        self.assertEqual(linReg.range, (-3.0, -1.0))
        
    def testVeryNegativeDomainRange(self):
        linReg = lrp.LinearRegression([-1000000000, -2000000000, -3000000000], [-1000000000, -2000000000, -3000000000])
        self.assertEqual(linReg.domain, (-3000000000.0, -1000000000))
        self.assertEqual(linReg.range, (-3000000000.0, -1000000000))
        
    def testSmallDomainRange(self):
        linReg = lrp.LinearRegression([.00000001, .00000002, .00000003], [.00000001, .00000002, .00000003])
        self.assertAlmostEqual(linReg.domain, (.00000001, .00000003))
        self.assertAlmostEqual(linReg.range, (.00000001, .00000003))
        
    def testLargeDomainRange(self):
        linReg = lrp.LinearRegression([1, 2, 1000000000], [2, 3, 4000000000])
        self.assertEqual(linReg.domain, (1.0, 1000000000.0))
        self.assertEqual(linReg.range, (2.0, 4000000000.0))
        
    def testLargePrecisionDomainRange(self):
        linReg = lrp.LinearRegression([1000000.0000001, 2000000.0000002, 3000000.0000003], 
                                      [1000000.0000001, 2000000.0000002, 3000000.0000003])
        self.assertEqual(linReg.domain, (1000000.0000001, 3000000.0000003))
        self.assertEqual(linReg.range, (1000000.0000001, 3000000.0000003))


if __name__ == "__main__":
    unittest.main()