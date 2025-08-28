The development history of POGEMA, starting from version 1.0.0.

Version 1.5.0 (August, 2025)
	•	Added support of custom targets_xy for lifelong MAPF.
	•	Improved work with rectangular grids. Added width and height attributes to GridConfig.
	•	Added method update_config to properly update all attributes of GridConfig.
	•	Added more tests for new features - custom targets_xy and width/height attributes.

Version 1.4.0 (April 5, 2025)
	•	Extended limits for size of maps and number of agents.
	•	Fixed ep_length value.
	•	Updated some tests.

Version 1.3.0 (June 13, 2024)

	•	Updates for integration with newer version of gymnasium.
	•	Refactored AgentsDensityWrapper for modularity and clarity.
	•	Introduced RuntimeMetricWrapper for runtime monitoring.
	•	Enhanced map generation methods and added new metrics like SOC_Makespan.
	•	Animation improvements for better visualization.

Version 1.2.2 (September 22, 2023)

	•	Implemented soft collision handling for agent interactions.
	•	Improved lifelong scenario seeding for consistent agent behavior.
	•	Enhanced metric logging for better integration with PyMARL.

Version 1.2.0 (August 30, 2023)

	•	Fixed import issues with Literal and animation issues.
	•	Improved visualizations, including grid lines and border toggles.

Version 1.1.0 (March 30, 2023)

	•	Updated dependencies for gymnasium and PettingZoo.
	•	Added an option to remove animation borders for cleaner outputs.
	•	Fixed animation bugs for stuck agents.

Version 1.0.0 (February 2023)

	•	Launched core features, including A* policy implementations* and CI/CD support.
	•	Introduced basic visualization and fixed animation bugs.

Post-Version Updates

	•	Adjusted the number of agents in setups.
	•	Updated package metadata for better compatibility.
	•	Addressed legacy issues and improved benchmark generation.

Version 1.1.6 (February 21, 2023)

	•	Fixed static animation issues and added grid object rendering.

Version 1.1.5 (December 28, 2022)

	•	Fixed Python 3.7 compatibility issues and added map registries for better management.
	•	Introduced an attrition metric.

Version 1.1.4 (November 18, 2022)

	•	Fixed flake8 warnings for improved code quality.

Version 1.1.3 (October 28, 2022)

	•	Corrected random seed initialization for PogemaLifeLong.
	•	Optimized animation behavior.

Version 1.1.2 (October 5, 2022)

	•	Upgraded SVG animations for better compression.

Version 1.1.1 (August 30, 2022)

	•	Added map_name attributes for clearer references.
	•	Implemented new observation types (MAPF, POMAPF) and enhanced metrics aggregation.

Version 1.0.x and Earlier

	•	Introduced cooperative reward wrappers and lifelong environment versions.
	•	Dropped Python 3.6 support and refined animation handling.

Version 1.0.3 (June 29, 2022)

	•	Fixed rendering issues for inactive agents.

Version 1.0.2 (June 27, 2022)

	•	Enhanced customization for agent and target positions.

Pre-1.0.2 Development (June 2022)

	•	Improved tests, refactored code, and removed unnecessary dependencies.
    •	Introduced the PogemaLifeLong class with target generation and metrics tailored for lifelong scenarios.
	•	Introduced customizable map rules and agent/target positions.
	•	Simplified installation by removing unnecessary dependencies.


Version 1.0.0 (March 31, 2022)

	•	Added predefined configurations for grid environments and improved visualization.
	•	Integrated PettingZoo support and enhanced usability with better examples.
	•	Introduced grid_config class for environment configuration and improved state management.
	•	Added methods for relative position observations and fixed PettingZoo compatibility.
	•	Documentation improvements for better user guidance.

