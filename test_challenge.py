import unittest
import numpy as np

# Adjust this import to match the name of your main challenge script
import challenge

class TestClockChallenge(unittest.TestCase):
    
    def test_reporter_baseline(self):
        """The unedited reporter should yield an objective of exactly 1.0 and zero edits."""
        result = objective(REPORTER_PARAMETERS)
        self.assertTrue(result["valid"], "Reporter baseline should be marked valid.")
        self.assertAlmostEqual(result["objective"], 1.0, places=4, 
                               msg="Baseline objective must be 1.0")
        self.assertEqual(result["number_genes_edited"], 0)
        self.assertEqual(result["score_improvement_percent"], 0.0)

    def test_parameter_validation_bounds(self):
        """validate_parameters should reject negative or zero values."""
        bad_p = REPORTER_PARAMETERS.copy()
        bad_p[0] = -1.0
        with self.assertRaises(ValueError):
            validate_parameters(bad_p)
            
        bad_p[0] = 0.0
        with self.assertRaises(ValueError):
            validate_parameters(bad_p)

    def test_parameter_validation_shape(self):
        """validate_parameters should reject arrays that do not have exactly 45 elements."""
        with self.assertRaises(ValueError):
            validate_parameters(REPORTER_PARAMETERS[:-1])  # 44 parameters
            
        with self.assertRaises(ValueError):
            validate_parameters(np.append(REPORTER_PARAMETERS, 1.0))  # 46 parameters

    def test_forbidden_edits_rejected(self):
        """Modifying a non-DNA parameter should flag the submission as invalid."""
        p_edit = REPORTER_PARAMETERS.copy()
        forbidden_idx = FORBIDDEN_INDICES[0]
        p_edit[forbidden_idx] *= 1.1  # Perturb by 10%
        
        result = objective(p_edit)
        self.assertFalse(result["valid"], "Modifying a forbidden parameter must invalidate the vector.")
        self.assertIn(PARAMETER_NAMES[forbidden_idx], result["forbidden_parameters"])
        self.assertIsNotNone(result["failure"])

    def test_edit_tracking(self):
        """Modifying an editable parameter should correctly identify the gene without flagging as forbidden."""
        p_edit = REPORTER_PARAMETERS.copy()
        editable_idx = EDITABLE_INDICES[0]
        p_edit[editable_idx] *= 1.05  # Perturb by 5%
        
        summary = edit_summary(p_edit)
        self.assertEqual(summary["number_forbidden_parameters"], 0)
        self.assertGreaterEqual(summary["number_genes_edited"], 1)
        self.assertEqual(summary["number_parameters_edited"], 1)

if __name__ == '__main__':
    unittest.main()