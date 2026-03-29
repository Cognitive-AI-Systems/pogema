# Changelog

## 2.0.0 (Upcoming)

- Removed PyMARL integration
- Added `PogemaWrapper` base class with explicit method forwarding
- `AnimationWrapper` with enable/disable/save API and zero overhead when inactive
- `soft` collision system as default
- Terminal render trimming

## 1.4.x (August 2025)

- Added support for custom `targets_xy` for lifelong MAPF.
- Improved work with rectangular grids by adding `width` and `height` attributes to `GridConfig`.
- Added `update_config` to properly update all attributes of `GridConfig`.
- Added more tests for `targets_xy` and `width`/`height` support.

## 1.4.0

_April 5, 2025_

- Extended limits for map size and number of agents.
- Fixed `ep_length`.
- Updated some tests.

## 1.3.0

_June 13, 2024_

- Updated integration for newer versions of Gymnasium.
- Refactored `AgentsDensityWrapper` for modularity and clarity.
- Introduced `RuntimeMetricWrapper` for runtime monitoring.
- Enhanced map generation methods and added new metrics such as `SOC_Makespan`.
- Improved animation visualization.

## 1.2.2

_September 22, 2023_

- Implemented soft collision handling for agent interactions.
- Improved lifelong scenario seeding for consistent agent behavior.
- Enhanced metric logging for better PyMARL integration.

## 1.2.0

_August 30, 2023_

- Fixed import issues with `Literal` and resolved animation issues.
- Improved visualizations, including grid lines and border toggles.

## 1.1.6

_February 21, 2023_

- Fixed static animation issues and added grid object rendering.

## Early 2023 Project Updates

- Launched core features, including A* policy implementations and CI/CD support.
- Introduced basic visualization and fixed animation bugs.
- Adjusted the number of agents in setups.
- Updated package metadata for better compatibility.
- Addressed legacy issues and improved benchmark generation.

## 1.1.5

_December 28, 2022_

- Fixed Python 3.7 compatibility issues and added map registries for better management.
- Introduced an attrition metric.

## 1.1.4

_November 18, 2022_

- Fixed `flake8` warnings for improved code quality.

## 1.1.3

_October 28, 2022_

- Corrected random seed initialization for `PogemaLifeLong`.
- Optimized animation behavior.

## 1.1.2

_October 5, 2022_

- Upgraded SVG animations for better compression.

## 1.1.1

_August 30, 2022_

- Added `map_name` attributes for clearer references.
- Implemented new observation types (`MAPF`, `POMAPF`) and enhanced metrics aggregation.

## 1.1.0

_August 11, 2022_

- Updated dependencies for Gymnasium and PettingZoo.
- Added an option to remove animation borders for cleaner outputs.
- Fixed animation bugs for stuck agents.

## Additional 1.0.x Updates

- Introduced cooperative reward wrappers and lifelong environment versions.
- Dropped Python 3.6 support and refined animation handling.

## 1.0.3

_June 29, 2022_

- Fixed rendering issues for inactive agents.

## 1.0.2

_June 27, 2022_

- Enhanced customization for agent and target positions.

## Pre-1.0.2 Development

_June 2022_

- Improved tests, refactored code, and removed unnecessary dependencies.
- Introduced the `PogemaLifeLong` class with target generation and metrics tailored for lifelong scenarios.
- Introduced customizable map rules and agent/target positions.
- Simplified installation by removing unnecessary dependencies.

## 1.0.0

_March 31, 2022_

- Added predefined configurations for grid environments and improved visualization.
- Integrated PettingZoo support and enhanced usability with better examples.
- Introduced `GridConfig` for environment configuration and improved state management.
- Added methods for relative position observations and fixed PettingZoo compatibility.
- Improved documentation for better user guidance.
