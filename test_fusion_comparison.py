"""
Fusion Testing and Comparison Script

This script tests the fusion mechanism by combining 2-3 inputs and comparing
the fusion results against individual modality test results.

Usage:
    python test_fusion_comparison.py
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests
import json
from typing import Dict, List, Tuple
import time


class FusionTester:
    """Test fusion mechanism by combining multiple modalities and comparing results."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
    
    def test_url(self, url: str) -> Dict:
        """Test URL detection."""
        try:
            response = requests.post(
                f"{self.base_url}/api/detect/url",
                json={"url": url},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_email(self, email_text: str) -> Dict:
        """Test email detection."""
        try:
            response = requests.post(
                f"{self.base_url}/api/detect/email",
                json={"email_text": email_text},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_sms(self, sms_text: str) -> Dict:
        """Test SMS detection."""
        try:
            response = requests.post(
                f"{self.base_url}/api/detect/sms",
                json={"sms_text": sms_text},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_fusion(self, url: str = None, email_text: str = None, 
                   sms_text: str = None, qr_url: str = None) -> Dict:
        """Test fusion with multiple modalities."""
        try:
            payload = {}
            if url:
                payload["url"] = url
            if email_text:
                payload["email_text"] = email_text
            if sms_text:
                payload["sms_text"] = sms_text
            if qr_url:
                payload["qr_url"] = qr_url
            
            response = requests.post(
                f"{self.base_url}/api/detect/fusion",
                json=payload,
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def compare_fusion_vs_individual(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Compare fusion results against individual modality results.
        
        Args:
            test_cases: List of test cases, each containing:
                - name: Test case name
                - url: URL to test (optional)
                - email_text: Email text to test (optional)
                - sms_text: SMS text to test (optional)
                - expected_label: Expected phishing label (optional)
        
        Returns:
            List of comparison results
        """
        comparison_results = []
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"Test Case: {test_case['name']}")
            print(f"{'='*60}")
            
            result = {
                "test_case": test_case["name"],
                "individual_results": {},
                "fusion_result": None,
                "comparison": {}
            }
            
            # Test individual modalities
            individual_probs = {}
            
            if "url" in test_case and test_case["url"]:
                print(f"\nTesting URL: {test_case['url']}")
                url_result = self.test_url(test_case["url"])
                result["individual_results"]["url"] = url_result
                if "phishing_probability" in url_result:
                    individual_probs["url"] = url_result["phishing_probability"]
                    print(f"  URL Result: {url_result.get('label')} (prob: {url_result.get('phishing_probability', 0):.3f})")
            
            if "email_text" in test_case and test_case["email_text"]:
                print(f"\nTesting Email: {test_case['email_text'][:50]}...")
                email_result = self.test_email(test_case["email_text"])
                result["individual_results"]["email"] = email_result
                if "phishing_probability" in email_result:
                    individual_probs["email"] = email_result["phishing_probability"]
                    print(f"  Email Result: {email_result.get('label')} (prob: {email_result.get('phishing_probability', 0):.3f})")
            
            if "sms_text" in test_case and test_case["sms_text"]:
                print(f"\nTesting SMS: {test_case['sms_text'][:50]}...")
                sms_result = self.test_sms(test_case["sms_text"])
                result["individual_results"]["sms"] = sms_result
                if "phishing_probability" in sms_result:
                    individual_probs["sms"] = sms_result["phishing_probability"]
                    print(f"  SMS Result: {sms_result.get('label')} (prob: {sms_result.get('phishing_probability', 0):.3f})")
            
            # Test fusion
            print(f"\nTesting Fusion with {len(individual_probs)} modalities")
            fusion_result = self.test_fusion(
                url=test_case.get("url"),
                email_text=test_case.get("email_text"),
                sms_text=test_case.get("sms_text")
            )
            result["fusion_result"] = fusion_result
            
            if "phishing_probability" in fusion_result:
                fusion_prob = fusion_result["phishing_probability"]
                fusion_label = fusion_result.get("final_label")
                print(f"  Fusion Result: {fusion_label} (prob: {fusion_prob:.3f})")
                
                # Compare fusion vs individual
                result["comparison"]["fusion_probability"] = fusion_prob
                result["comparison"]["fusion_label"] = fusion_label
                result["comparison"]["individual_probabilities"] = individual_probs
                
                # Calculate agreement metrics
                individual_labels = []
                for modality, prob in individual_probs.items():
                    label = "Phishing" if prob >= 0.5 else "Legitimate"
                    individual_labels.append(label)
                
                if individual_labels:
                    agreement_count = sum(1 for label in individual_labels if label == fusion_label)
                    agreement_rate = agreement_count / len(individual_labels)
                    result["comparison"]["agreement_rate"] = agreement_rate
                    print(f"  Agreement with individual modalities: {agreement_rate:.2%}")
                    
                    # Check if fusion matches majority
                    if individual_labels:
                        majority_label = max(set(individual_labels), key=individual_labels.count)
                        result["comparison"]["matches_majority"] = (fusion_label == majority_label)
                        print(f"  Matches majority: {result['comparison']['matches_majority']}")
            
            comparison_results.append(result)
        
        return comparison_results
    
    def generate_report(self, comparison_results: List[Dict]) -> str:
        """Generate a comparison report."""
        report = []
        report.append("=" * 80)
        report.append("FUSION VS INDIVIDUAL MODALITY COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        for result in comparison_results:
            report.append(f"Test Case: {result['test_case']}")
            report.append("-" * 80)
            
            # Individual results
            report.append("\nIndividual Modality Results:")
            for modality, modality_result in result["individual_results"].items():
                if "phishing_probability" in modality_result:
                    prob = modality_result["phishing_probability"]
                    label = modality_result.get("label", "Unknown")
                    report.append(f"  {modality.upper()}: {label} (probability: {prob:.4f})")
                else:
                    report.append(f"  {modality.upper()}: Error - {modality_result.get('error', 'Unknown error')}")
            
            # Fusion result
            report.append("\nFusion Result:")
            if result["fusion_result"] and "phishing_probability" in result["fusion_result"]:
                fusion_prob = result["fusion_result"]["phishing_probability"]
                fusion_label = result["fusion_result"].get("final_label", "Unknown")
                report.append(f"  Label: {fusion_label}")
                report.append(f"  Probability: {fusion_prob:.4f}")
            else:
                report.append(f"  Error - {result['fusion_result'].get('error', 'Unknown error')}")
            
            # Comparison
            if result["comparison"]:
                report.append("\nComparison Metrics:")
                report.append(f"  Agreement Rate: {result['comparison'].get('agreement_rate', 0):.2%}")
                report.append(f"  Matches Majority: {result['comparison'].get('matches_majority', False)}")
            
            report.append("")
        
        # Summary statistics
        report.append("=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        total_tests = len(comparison_results)
        fusion_matches_majority = sum(1 for r in comparison_results 
                                     if r["comparison"].get("matches_majority", False))
        avg_agreement = sum(r["comparison"].get("agreement_rate", 0) 
                           for r in comparison_results) / total_tests if total_tests > 0 else 0
        
        report.append(f"Total Test Cases: {total_tests}")
        report.append(f"Fusion Matches Majority: {fusion_matches_majority}/{total_tests} ({fusion_matches_majority/total_tests:.2%})")
        report.append(f"Average Agreement Rate: {avg_agreement:.2%}")
        
        return "\n".join(report)


def main():
    """Main testing function."""
    print("Fusion Testing and Comparison Script")
    print("=" * 60)
    
    # Initialize tester
    tester = FusionTester()
    
    # Define test cases with 2-3 modalities
    test_cases = [
        {
            "name": "Phishing URL + Phishing Email",
            "url": "http://secure-login-verify-account.com/login",
            "email_text": "Dear Customer, Your account will be suspended unless you verify your identity immediately. Click here to verify: http://secure-login-verify-account.com/login"
        },
        {
            "name": "Legitimate URL + Legitimate Email",
            "url": "https://github.com",
            "email_text": "Thank you for your purchase. Your order has been confirmed and will be shipped within 2-3 business days."
        },
        {
            "name": "Phishing URL + Legitimate SMS",
            "url": "http://banking-secure-update.com/verify",
            "sms_text": "Your package has been delivered. Thank you for using our service."
        },
        {
            "name": "Phishing Email + Phishing SMS",
            "email_text": "URGENT: Your account has been compromised. Click here to reset your password immediately.",
            "sms_text": "URGENT: Your account will be closed unless you act now. Call this number to verify."
        },
        {
            "name": "Mixed Signals (Phishing URL + Legitimate Email)",
            "url": "http://phishing-site.com/login",
            "email_text": "Your monthly statement is now available. Log in to your account to view it."
        },
        {
            "name": "Three Modalities - All Phishing",
            "url": "http://fake-banking-verify.com",
            "email_text": "Security Alert: Verify your account immediately or it will be suspended.",
            "sms_text": "Your account has been limited. Click to restore access now."
        },
        {
            "name": "Three Modalities - All Legitimate",
            "url": "https://www.microsoft.com",
            "email_text": "Thank you for subscribing to our newsletter.",
            "sms_text": "Your appointment is confirmed for tomorrow at 2 PM."
        },
        {
            "name": "Three Modalities - Mixed Signals",
            "url": "http://suspicious-site.com",
            "email_text": "Your order has been shipped successfully.",
            "sms_text": "URGENT: Account security alert. Verify now."
        }
    ]
    
    # Run comparison tests
    print(f"\nRunning {len(test_cases)} test cases...")
    comparison_results = tester.compare_fusion_vs_individual(test_cases)
    
    # Generate and display report
    report = tester.generate_report(comparison_results)
    print("\n" + report)
    
    # Save report to file
    report_file = BASE_DIR / "fusion_comparison_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Save detailed results as JSON
    results_file = BASE_DIR / "fusion_comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
